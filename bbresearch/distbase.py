"""評価の土台（候補 1、2026-09-28 オーナー決定）: ボラだけの基準と、向きの上乗せの測り方。

行 t は足 t（[t, t+1h)）の終値の時点。目的変数 y_t = log(close_{t+h} / close_t)（h = 4, 24）。
比べる分布（すべて「重み付きの標本」V（行ごとに昇順）と重み W で表す）:
  unconditional   学習期間の y の経験分布
  rolling{n}      その時点で確定した直近 n 個の y の経験分布
  har             HAR 型: log(次の h 本の RV の和) を log(直近 1・4・24・168 本の RV の平均) と時刻・曜日のダミーに OLS で回帰し、
                  σ̂_t = exp(予測 / 2)。分布は σ̂_t × z（z は学習期間の y / σ̂ の経験分布。Dudek ほか 2025 の残差の方法）
  garch           GARCH(1,1)-t を 1 時間の対数リターンに当てはめ、h 本先までの条件付き分散の和の平方根を σ̂_t とする。分布は har と同じ形
  forest          分位点回帰フォレスト（分位点ビン分割、1 時間足の指標 373 個。scripts/dist_forecast.py の 4 回目で最良の設定）
  forest_fixed    forest の予測分布を、行ごとの中央値が学習期間の y の中央値になるよう平行移動したもの（幅と形はそのまま、向きの情報を消す）
  sign_magnitude  符号と大きさの分解（Brou & Luger 2026 を応用）: 大きさは har（|z| の分位点 K 点 × σ̂_t）、符号は
                  「特徴量と大きさ」から LightGBM で学習した P(y > 0 | |z|, x)
RV_t は足 t の 1 分ごとの対数リターンの二乗和（1 分の終値は約定のない分を直前の値で埋める）。
すべて行 t の時点で分かるデータだけを使う（tests/test_distbase.py で未来を書き換えても過去が変わらないことを確かめる）。
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from bbresearch.indicators import forward_return, indicator_table
from bbresearch.qrf import QuantileForest

MIN_MS = 60_000
CHUNK = 512
MODELS = ("unconditional", "rolling168", "rolling720", "garch", "har", "forest", "forest_fixed", "sign_magnitude")
LABELS = {"unconditional": "無条件", "rolling168": "直近 168 本", "rolling720": "直近 720 本", "garch": "GARCH-t 型",
          "har": "HAR 型", "forest": "フォレスト", "forest_fixed": "フォレスト（位置を固定）", "sign_magnitude": "符号と大きさ"}


# ------------------------------------------------------------------ 1 分足と実現分散

def minute_close_from_raw(root, pair: str, start, end, chunk_days: int = 7) -> tuple[pd.Series, pd.Series]:
    """保存済みの約定から、1 分ごとの最後の約定価格（約定のない分は NaN）と約定の件数。index は分の開始時刻（UTC）。"""
    from bbdata.download import load_transactions, to_utc

    start, end = to_utc(start), to_utc(end)
    lasts, counts = [], []
    cur = start
    while cur < end:
        nxt = min(cur + pd.Timedelta(days=chunk_days), end)
        t = load_transactions(root, pair, cur, nxt)
        n = int((nxt - cur) / pd.Timedelta(minutes=1))
        k = (t["executed_at"].to_numpy("int64") - cur.value // 1_000_000) // MIN_MS
        px = t["price"].to_numpy(float)
        last = np.full(n, np.nan)
        if len(k):
            ends = np.searchsorted(k, np.arange(n), side="right") - 1
            has = np.bincount(k, minlength=n)[:n] > 0
            last[has] = px[ends[has]]
        lasts.append(last)
        counts.append(np.bincount(k, minlength=n)[:n] if len(k) else np.zeros(n, int))
        cur = nxt
    idx = pd.date_range(start, end, freq="1min", inclusive="left")
    return pd.Series(np.concatenate(lasts), index=idx), pd.Series(np.concatenate(counts), index=idx)


def hourly_rv(minute_close: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """足 t（[t, t+1h)）の 1 分ごとの対数リターンの二乗和。分 m のリターンは m の終値と m−1 の終値の対数比で、m を含む足に入れる。"""
    lp = np.log(minute_close.ffill())
    r2 = lp.diff() ** 2
    rv = r2.groupby(r2.index.floor("h")).sum(min_count=1)
    return rv.reindex(index)


# ------------------------------------------------------------------ HAR 型

def har_features(rv: pd.Series, windows=(1, 4, 24, 168), floor: float = 1e-10, calendar: bool = True) -> pd.DataFrame:
    """行 t: log(足 t を含む直近 k 本の RV の平均)（k = windows）と、時刻（1〜23 時）・曜日（火〜日）のダミー。"""
    rvf = rv.clip(lower=floor)
    cols = {f"har_{k}": np.log(rvf.rolling(k, min_periods=k).mean()) for k in windows}
    if calendar:
        hour, wd = rv.index.hour, rv.index.dayofweek
        for k in range(1, 24):
            cols[f"hour_{k}"] = (hour == k).astype(float)
        for k in range(1, 7):
            cols[f"wd_{k}"] = (wd == k).astype(float)
    return pd.DataFrame(cols, index=rv.index)


def har_target(rv: pd.Series, h: int, floor: float = 1e-10) -> pd.Series:
    """行 t: log(RV_{t+1} + … + RV_{t+h})。"""
    return np.log(rv.clip(lower=floor).rolling(h, min_periods=h).sum().shift(-h))


class HARVol:
    """log(次の h 本の RV の和) の OLS。σ̂ = exp(予測 / 2)。"""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HARVol":
        A = np.column_stack([np.ones(len(X)), X])
        self.coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ self.coef
        self.r2 = float(1 - resid.var() / y.var())
        return self

    def sigma(self, X: np.ndarray) -> np.ndarray:
        return np.exp(0.5 * (np.column_stack([np.ones(len(X)), X]) @ self.coef))


# ------------------------------------------------------------------ GARCH(1,1)-t

def fit_garch_t(r: np.ndarray) -> dict:
    """1 時間の対数リターン（学習期間）に GARCH(1,1)-t（平均は定数）を当てはめる。100 倍して推定し、元の尺度で返す。"""
    from arch import arch_model

    res = arch_model(100 * np.asarray(r, float), mean="Constant", vol="GARCH", p=1, q=1, dist="t", rescale=False).fit(disp="off")
    p = res.params
    return {"mu": float(p["mu"] / 100), "omega": float(p["omega"] / 1e4), "alpha": float(p["alpha[1]"]),
            "beta": float(p["beta[1]"]), "nu": float(p["nu"]), "loglik": float(res.loglikelihood)}


def garch_next_var(par: dict, r: np.ndarray) -> np.ndarray:
    """s[t] = σ²_{t+1|t}（r_t までで決まる次の 1 時間の条件付き分散）。初期値は無条件分散。r の NaN は平均とみなす。"""
    om, a, b, mu = par["omega"], par["alpha"], par["beta"], par["mu"]
    e2 = np.nan_to_num(np.asarray(r, float) - mu) ** 2
    out = np.empty(len(e2))
    prev = om / max(1e-12, 1 - a - b)
    for t in range(len(e2)):
        prev = om + a * e2[t] + b * prev
        out[t] = prev
    return out


def garch_sigma_h(par: dict, next_var: np.ndarray, h: int) -> np.ndarray:
    """Σ_{k=1..h} E_t[σ²_{t+k}] の平方根。E_t[σ²_{t+k}] = σ̄² + φ^{k−1}(σ²_{t+1|t} − σ̄²)、φ = α + β。"""
    phi = par["alpha"] + par["beta"]
    vbar = par["omega"] / max(1e-12, 1 - phi)
    geo = h if abs(1 - phi) < 1e-12 else (1 - phi**h) / (1 - phi)
    return np.sqrt(np.clip(h * vbar + (next_var - vbar) * geo, 1e-18, None))


# ------------------------------------------------------------------ 重み付き標本の分布の評価

def crps_rows(V: np.ndarray, W: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """行ごとの重み付き標本（V は行ごとに昇順）の CRPS: E|Y − y| − ½ E|Y − Y'|。V・W は (1, n) でもよい。"""
    W = W / W.sum(axis=1, keepdims=True)
    term1 = (W * np.abs(V - obs[:, None])).sum(axis=1)
    cw = np.cumsum(W, axis=1)
    e_abs = 2.0 * (W * V * (2 * cw - W - 1.0)).sum(axis=1)   # before − after = (cw − W) − (1 − cw)
    return term1 - 0.5 * e_abs


def dist_metrics(V: np.ndarray, W: np.ndarray, obs: np.ndarray, qs, cs, tw: float) -> dict[str, np.ndarray]:
    """分位点・CRPS・P(Y > c)・平均・PIT・閾値で重みを付けた CRPS（上側 w = 1{z > tw}、下側 w = 1{z < −tw}）。

    閾値の重みの CRPS は、標本と観測を v(z) = max(z, tw)（下側は min(z, −tw)）で変換した CRPS（Allen ほか 2023）。"""
    rows = len(obs)
    Wb = np.broadcast_to(W, (rows, W.shape[1])) if W.shape[0] == 1 else W
    Wb = Wb / Wb.sum(axis=1, keepdims=True)
    Vb = np.broadcast_to(V, (rows, V.shape[1])) if V.shape[0] == 1 else V
    cw = np.cumsum(Wb, axis=1)
    out = {"q": np.empty((rows, len(qs))), "p": np.empty((rows, len(cs)))}
    r = np.arange(rows)
    for j, q in enumerate(qs):
        idx = np.minimum((cw < q - 1e-12).sum(axis=1), Vb.shape[1] - 1)
        out["q"][:, j] = Vb[r, idx]
    out["crps"] = crps_rows(V, Wb, obs)
    for j, c in enumerate(cs):
        out["p"][:, j] = (Wb * (Vb > c)).sum(axis=1)
    out["mean"] = (Wb * Vb).sum(axis=1)
    out["pit"] = (Wb * (Vb < obs[:, None])).sum(axis=1) + 0.5 * (Wb * (Vb == obs[:, None])).sum(axis=1)
    out["tw_up"] = crps_rows(np.maximum(V, tw), Wb, np.maximum(obs, tw))
    out["tw_dn"] = crps_rows(np.minimum(V, -tw), Wb, np.minimum(obs, -tw))
    return out


def weighted_median(V: np.ndarray, W: np.ndarray) -> np.ndarray:
    Wn = W / W.sum(axis=1, keepdims=True)
    cw = np.cumsum(Wn, axis=1)
    idx = np.minimum((cw < 0.5 - 1e-12).sum(axis=1), V.shape[1] - 1)
    Vb = np.broadcast_to(V, Wn.shape) if V.shape[0] == 1 else V
    return Vb[np.arange(len(W)), idx]


def pinball_rows(obs: np.ndarray, q: np.ndarray, qs) -> np.ndarray:
    """行ごとの、分位点のピンボール損失の平均。"""
    d = obs[:, None] - q
    tau = np.asarray(qs)[None, :]
    return np.maximum(tau * d, (tau - 1) * d).mean(axis=1)


def brier(p: np.ndarray, o: np.ndarray) -> np.ndarray:
    return (p - o) ** 2


def logloss(p: np.ndarray, o: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(o * np.log(p) + (1 - o) * np.log(1 - p))


def corp(p: np.ndarray, o: np.ndarray) -> dict:
    """ブライアスコアの CORP 分解（Dimitriadis ほか 2021）: 平均 = MCB − DSC + UNC。較正し直した予測は PAV（単調回帰）。"""
    pr = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, o).predict(p)
    s, s_rc, s_clim = brier(p, o).mean(), brier(pr, o).mean(), brier(np.full(len(o), o.mean()), o).mean()
    return {"brier": float(s), "mcb": float(s - s_rc), "dsc": float(s_clim - s_rc), "unc": float(s_clim)}


def elementary_payoff(mean: np.ndarray, y: np.ndarray, theta: float) -> np.ndarray:
    """「予測平均が θ を超えたら買い、h 本後に売る」の、判断 1 回あたりの費用 θ 控除後の損益（買わない回は 0）。

    平均の初等スコア S_θ(x, y) = (y − θ)_+ − 1{x > θ}(y − θ)（Ehm ほか 2016）なので、2 つの予測の S_θ の差は、この損益の差の符号を変えたもの。"""
    return np.where(mean > theta, y - theta, 0.0)


# ------------------------------------------------------------------ 分布ごとの標本

def scaled_dist(z_sorted: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return sigma[:, None] * z_sorted[None, :], np.full((1, len(z_sorted)), 1.0 / len(z_sorted))


def rolling_windows(y_all: np.ndarray, pos: np.ndarray, n: int, h: int) -> np.ndarray:
    """位置 pos の行の時点で確定した直近 n 個の y（位置 pos − h − n 〜 pos − h − 1）を昇順に。y_all は 1 時間ごとに抜けなく並んだ y。

    位置 j の y は足 j + h の終値で確定する。scripts/dist_forecast.py の直近 n 本（qrf.rolling_empirical）と同じく、
    ちょうど確定した位置 pos − h は入れない。"""
    from numpy.lib.stride_tricks import sliding_window_view

    win = sliding_window_view(y_all, n)
    return np.sort(win[pos - h - n], axis=1)


def sm_dist(p: np.ndarray, grid: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """符号と大きさ: 大きさ g_k（昇順、等しい重み 1/K）と P(y > 0 | g_k) から、±σ̂ g_k の重み付き標本を作る（昇順）。"""
    K = len(grid)
    V = sigma[:, None] * np.concatenate([-grid[::-1], grid])[None, :]
    W = np.concatenate([(1 - p[:, ::-1]) / K, p / K], axis=1)
    return V, W


# ------------------------------------------------------------------ 学習と評価（実データと合成データで同じ手順）

def _lgbm(params: dict, n_jobs: int):
    import lightgbm as lgb

    return lgb.LGBMClassifier(**params, n_jobs=n_jobs, verbose=-1)


def fit_sign_classifier(X: np.ndarray, y: np.ndarray, h: int, smc: dict, n_jobs: int):
    """符号の分類器。木の数は学習期間の末尾 valid_frac で早期停止して決め（境界の h 行は除く）、学習期間全体で学習し直す。

    木の数を固定すると、向きの情報がないデータでも確率が極端になり、ブライアスコアが大きく悪化した（合成データで確認）。"""
    import lightgbm as lgb

    cut = len(y) - int(len(y) * float(smc["valid_frac"]))
    stop = [lgb.early_stopping(int(smc["early_stopping_rounds"]), verbose=False)]
    try:        # LightGBM 4.6 以降は eval_X / eval_y（eval_set は非推奨）
        probe = _lgbm(smc["lgbm"], n_jobs).fit(X[:cut - h], y[:cut - h], eval_X=(X[cut:],), eval_y=(y[cut:],), callbacks=stop)
    except TypeError:
        probe = _lgbm(smc["lgbm"], n_jobs).fit(X[:cut - h], y[:cut - h], eval_set=[(X[cut:], y[cut:])], callbacks=stop)
    best = max(1, int(probe.best_iteration_ or smc["lgbm"]["n_estimators"]))
    return _lgbm(smc["lgbm"] | {"n_estimators": best}, n_jobs).fit(X, y), best


def run_pipeline(h1: pd.DataFrame, minute_close: pd.Series, cfg: dict, feats: pd.DataFrame | None = None,
                 n_jobs: int = -1, log=print) -> dict:
    """1 時間足（OHLCV）と 1 分の終値から、ホライズンごとに全モデルを学習期間で 1 回学習し、評価期間の行ごとの値を返す。"""
    qs, cs, tw = tuple(cfg["quantiles"]), tuple(cfg["thresholds"]), float(cfg["cost"])
    split = pd.Timestamp(cfg["split"], tz="UTC")
    hc, fc, smc = cfg["har"], cfg["forest"], cfg["sign_magnitude"]
    if feats is None:
        feats = indicator_table(h1)
    close = h1["close"]
    if len(h1) and (h1.index[-1] - h1.index[0]) != pd.Timedelta(hours=len(h1) - 1):
        raise ValueError("1 時間足に抜けがある（直近 n 本の位置と GARCH の時間がずれる）")
    rv = hourly_rv(minute_close, h1.index)
    X_har_all = har_features(rv, tuple(hc["windows"]), float(hc["rv_floor"]), bool(hc["calendar"]))
    r1 = np.log(close).diff()
    r_train = r1[(r1.index < split)].dropna().to_numpy()
    gpar = fit_garch_t(r_train)
    next_var = pd.Series(garch_next_var(gpar, r1.to_numpy()), index=h1.index)
    log(f"GARCH-t: ω {gpar['omega']:.3e}、α {gpar['alpha']:.3f}、β {gpar['beta']:.3f}、ν {gpar['nu']:.2f}")
    out = {"garch": gpar, "horizons": {}}
    for h in cfg["horizons"]:
        h = int(h)
        y = forward_return(close, h).rename("y")
        yv = har_target(rv, h, float(hc["rv_floor"])).rename("yv")
        data = feats.join(X_har_all).join(y).join(yv).dropna()
        train = data[data.index < split - pd.Timedelta(hours=h)]
        test = data[data.index >= split]
        fcols, hcols = list(feats.columns), list(X_har_all.columns)
        ytr, yte = train["y"].to_numpy(), test["y"].to_numpy()
        # HAR 型
        har = HARVol().fit(train[hcols].to_numpy(), train["yv"].to_numpy())
        s_har_tr, s_har_te = har.sigma(train[hcols].to_numpy()), har.sigma(test[hcols].to_numpy())
        z_har = np.sort(ytr / s_har_tr)
        # GARCH-t 型
        s_g = pd.Series(garch_sigma_h(gpar, next_var.to_numpy(), h), index=h1.index)
        s_g_tr, s_g_te = s_g.reindex(train.index).to_numpy(), s_g.reindex(test.index).to_numpy()
        z_g = np.sort(ytr / s_g_tr)
        # フォレスト
        qf = QuantileForest(int(fc["n_estimators"]), int(fc["min_samples_leaf"]), fc["max_features"], float(fc["max_samples"]),
                            random_state=int(fc.get("random_state", 0)), n_jobs=n_jobs, split_target=fc["split_target"],
                            n_bins=int(fc["n_bins"])).fit(train[fcols].to_numpy(), ytr)
        m0 = float(np.median(ytr))
        # 符号と大きさ
        mag_tr = np.abs(ytr) / s_har_tr
        K = int(smc["grid"])
        grid = np.quantile(mag_tr, (np.arange(K) + 0.5) / K)
        clf, sm_iter = fit_sign_classifier(np.column_stack([train[fcols].to_numpy(), mag_tr]), (ytr > 0).astype(int), h, smc,
                                           n_jobs if n_jobs > 0 else 4)
        # 直近 n 本（特徴量の有無によらず 1 時間ごとに並んだ y から、確定した値だけを使う）
        y_all = y.to_numpy()
        pos_test = h1.index.get_indexer(test.index)
        y_uncond = np.sort(ytr)
        res = {m: {} for m in MODELS}
        Xte = test[fcols].to_numpy()
        for s in range(0, len(test), CHUNK):
            e = min(s + CHUNK, len(test))
            obs = yte[s:e]
            parts = {}
            parts["unconditional"] = (y_uncond[None, :], np.full((1, len(y_uncond)), 1.0 / len(y_uncond)))
            for n in cfg["rolling_n"]:
                win = rolling_windows(y_all, pos_test[s:e], int(n), h)
                parts[f"rolling{n}"] = (win, np.full((1, int(n)), 1.0 / int(n)))
            parts["har"] = scaled_dist(z_har, s_har_te[s:e])
            parts["garch"] = scaled_dist(z_g, s_g_te[s:e])
            W = qf.weights(Xte[s:e])
            Vf = qf._y[None, :]
            parts["forest"] = (Vf, W)
            parts["forest_fixed"] = (Vf + (m0 - weighted_median(Vf, W))[:, None], W)
            Xg = np.column_stack([np.repeat(Xte[s:e], K, axis=0), np.tile(grid, e - s)])
            p_up = clf.predict_proba(Xg)[:, 1].reshape(e - s, K)
            parts["sign_magnitude"] = sm_dist(p_up, grid, s_har_te[s:e])
            for m, (V, Wm) in parts.items():
                for k, v in dist_metrics(V, Wm, obs, qs, cs, tw).items():
                    res[m].setdefault(k, []).append(v)
        for m in res:
            res[m] = {k: np.concatenate(v) for k, v in res[m].items()}
        out["horizons"][h] = {"index": test.index, "y": yte, "models": res, "n_train": len(train),
                              "train_start": str(train.index[0]), "train_end": str(train.index[-1]),
                              "har_coef": dict(zip(["const"] + hcols, har.coef.round(6).tolist())), "har_r2": har.r2,
                              "median_train": m0, "z_har_q": np.quantile(z_har, qs).tolist(), "sm_trees": sm_iter}
        log(f"h = {h}: 学習 {len(train):,} 行、評価 {len(test):,} 行、HAR の R² {har.r2:.3f}、符号の分類器の木 {sm_iter}")
    return out


# ------------------------------------------------------------------ 行ごとの損失と集計

def losses(res: dict, y: np.ndarray, cfg: dict) -> dict[str, np.ndarray]:
    """比較に使う行ごとの損失（小さいほど良い）。"""
    qs, cs = tuple(cfg["quantiles"]), tuple(cfg["thresholds"])
    out = {"crps": res["crps"], "pinball": pinball_rows(y, res["q"], qs), "tw_up": res["tw_up"], "tw_dn": res["tw_dn"],
           "pinball50": pinball_rows(y, res["q"][:, [qs.index(0.5)]], (0.5,))}
    eps = float(cfg["logloss_eps"])
    for j, c in enumerate(cs):
        o = (y > c).astype(float)
        out[f"brier_{j}"] = brier(res["p"][:, j], o)
        out[f"logloss_{j}"] = logloss(res["p"][:, j], o, eps)
    # 初等スコアの差 = 損益の差の符号違い。損失として「−損益」を使う
    out["neg_payoff"] = -elementary_payoff(res["mean"], y, float(cfg["elementary_theta"]))
    return out


def summary_row(res: dict, y: np.ndarray, cfg: dict) -> dict:
    qs, cs = tuple(cfg["quantiles"]), tuple(cfg["thresholds"])
    L = losses(res, y, cfg)
    lo, hi = qs.index(0.05), qs.index(0.95)
    row = {k: float(np.mean(v)) for k, v in L.items()}
    row["coverage"] = {str(q): float(np.mean(y <= res["q"][:, j])) for j, q in enumerate(qs)}
    row["interval90"] = float(np.mean((y >= res["q"][:, lo]) & (y <= res["q"][:, hi])))
    row["width90_median"] = float(np.median(res["q"][:, hi] - res["q"][:, lo]))
    for j, c in enumerate(cs):
        o = (y > c).astype(int)
        p = res["p"][:, j]
        row[f"auc_{j}"] = float(roc_auc_score(o, p)) if 0 < o.sum() < len(o) and np.ptp(p) > 0 else 0.5
        row[f"corp_{j}"] = corp(p, o.astype(float))
    row["payoff"] = -row.pop("neg_payoff")
    row["buy_share"] = float(np.mean(res["mean"] > float(cfg["elementary_theta"])))
    row["murphy"] = {str(th): float(np.mean(elementary_payoff(res["mean"], y, th))) for th in cfg["murphy_thetas"]}
    # 予測平均の上位・下位の群の実現リターンの差（群は評価期間全体の分位で決める）
    share = float(cfg["top_share"])
    lo_c, hi_c = np.quantile(res["mean"], [share, 1 - share])
    top, bot = res["mean"] >= hi_c, res["mean"] <= lo_c
    if top.sum() and bot.sum() and np.ptp(res["mean"]) > 0:
        row["spread_series"] = y * (top / top.mean() - bot / bot.mean())
    return row


def pit_subseries(pit: np.ndarray, h: int) -> dict:
    """PIT を t mod h の h 本の部分系列に分け、それぞれの一様性を KS 検定。最小の p 値とボンフェローニ補正した値。"""
    from scipy.stats import kstest

    ps = [float(kstest(pit[k::h], "uniform").pvalue) for k in range(h) if len(pit[k::h]) > 10]
    return {"min_p": min(ps), "bonferroni": min(1.0, min(ps) * len(ps)), "n_sub": len(ps)}


def rank_percentile(x: float, null: list[float]) -> float:
    """帰無の監査の p 値: (1 + 帰無の値が x 以上の回数) / (1 + 回数)。"""
    a = np.asarray([v for v in null if np.isfinite(v)])
    return float((1 + np.sum(a >= x)) / (1 + len(a))) if len(a) else float("nan")


def year_means(index: pd.DatetimeIndex, v: np.ndarray) -> dict[int, float]:
    s = pd.Series(v, index=index)
    return {int(k): float(x) for k, x in s.groupby(s.index.year).mean().items()}


def safe(x):
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else x
