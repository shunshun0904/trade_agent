"""h 分後に上がるか下がるかを予測する二段構えのモデル（2026-09-25 オーナー指示。h は spec の horizon_min）。

    python -m bbresearch direction --config configs/research.yaml --spec configs/direction.yaml --out reports/direction

- モデル B（補強用）: 1 分ごとに、h 分後の価格が今より高いかを予測する。
- モデル A（判断用）: h 分ごと（h = 15 なら足の区切り、60 なら毎正時）に同じことを予測する。入力は B と同じ
  特徴量に、直近 h 分の B の予測の要約（平均・最新・変化・ばらつき・直近 1/4 の平均）を加えたもの
  （スタッキング、A+B）。B の予測は、その時点のデータで学習していないモデルの予測だけを使う（学習期間は
  Purged K-fold の out-of-fold、評価期間は学習期間だけで学習した B の予測）。
- A+B+L（spec の level_summary: true のとき）: さらに、直近 h 分の生の価格帯別出来高・TPO の水準
  （PROFILE_COLUMNS）の平均と変化（最新 − h 分前）を加える（2026-09-26 オーナー指示: 1 時間の判断に
  1 分ごとの TPO×価格帯別出来高の推移を使う）。

特徴量（オーナー決定: 全部）。どれも時刻 t より前のデータだけから作る:
- 1 分足から: 直近 1・5・15・60 分の対数リターン、1・5・15 分の出来高・売買の偏り・約定件数
- 15 分足から（t 以前に確定した最新の足）: features.bar_features と distfeat.dist_bar_features
- 価格帯別出来高・TPO（直近 24 時間、刻み 0.25σ）: POC・VAH・VAL までの距離（σ 単位）、バリューエリア内の
  位置、現在の価格帯の厚さ、上下 1σ 以内の量の割合、歪度、TPO のシングルプリントの割合
- 時刻・曜日

正解: 15 分後（t + 15 分の直前の約定価格）が t の直前の約定価格より高ければ 1、低ければ 0。同じなら除く。

期間（オーナー決定）: 学習・交差検証は [data.start, train_end)、評価は [train_end, data.end) で 1 回だけ。
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from bbdata.download import to_utc

from .cv import PurgedKFold
from .minute_features import (BAR_MS, BIN_SIGMA, FINE, H_MS, LOG_RANGE, MIN_MS, PROFILE_COLUMNS, WARMUP_HOURS,  # noqa: F401
                              feature_table, flow_features, minute_bars, minute_features, minute_grid,
                              profile_chunk, profile_features, taker_round_trip, targets)
from .model import metrics, params_hash
from .pipeline import barrier_params, load_market

log = logging.getLogger(__name__)
# ---------------------------------------------------------------- モデル

def lgbm(n_rows: int, seed: int = 0, n_jobs: int = 4):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=max(50, n_rows // 2000), subsample=0.5, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=seed, n_jobs=n_jobs, verbose=-1)


def oof_and_final(X: pd.DataFrame, y: np.ndarray, t_ms: np.ndarray, X_test: pd.DataFrame, n_splits: int,
                  horizon_ms: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """学習期間の out-of-fold 予測、学習期間全体で学習したモデルの評価期間の予測、fold ごとの指標。"""
    t0 = pd.Series(pd.to_datetime(t_ms, unit="ms", utc=True))
    tx = t0 + pd.Timedelta(milliseconds=horizon_ms)
    cv = PurgedKFold(n_splits=n_splits, embargo=pd.Timedelta(milliseconds=horizon_ms))
    oof = np.full(len(X), np.nan)
    folds = []
    for k, (tr, te) in enumerate(cv.split(t0, tx)):
        m = lgbm(len(tr), seed).fit(X.iloc[tr], y[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
        folds.append({"fold": k, **{kk: v for kk, v in metrics(y[te], oof[te]).items() if kk != "calibration"}})
    final = lgbm(len(X), seed).fit(X, y)
    return oof, final.predict_proba(X_test)[:, 1], folds, final


def stack_features(pred_min: pd.Series, t_ms: np.ndarray, n_min: int = 15) -> pd.DataFrame:
    """判断時刻 t に、直前 n_min 分（t − (n_min−1) 分 〜 t）の B の予測の要約を付ける。"""
    s = pred_min.reindex(np.concatenate([t_ms - k * MIN_MS for k in range(n_min - 1, -1, -1)]))
    v = s.to_numpy().reshape(n_min, len(t_ms)).T  # 行: 時刻、列: t−(n_min−1) … t
    q = max(1, n_min // 4)
    with np.errstate(invalid="ignore"):
        return pd.DataFrame({"B_mean": np.nanmean(v, axis=1), "B_last": v[:, -1], "B_slope": v[:, -1] - v[:, 0],
                             "B_std": np.nanstd(v, axis=1), "B_mean_recent": np.nanmean(v[:, -q:], axis=1)},
                            index=t_ms)


def level_summary(levels: pd.DataFrame, t_ms: np.ndarray, n_min: int) -> pd.DataFrame:
    """判断時刻 t に、直前 n_min 分（t − (n_min−1) 分 〜 t）の生の水準（1 分ごと）の平均と変化（最新 − 最初）を付ける。

    levels の index は 1 分ごとの時刻（ミリ秒）。窓に含まれない時刻は NaN として扱う。
    """
    cols = list(levels.columns)
    idx = np.concatenate([t_ms - k * MIN_MS for k in range(n_min - 1, -1, -1)])
    v = levels.reindex(idx).to_numpy(dtype=float).reshape(n_min, len(t_ms), len(cols))  # (窓, 時刻, 列)
    out = {}
    with np.errstate(invalid="ignore"):
        for j, c in enumerate(cols):
            out[f"L_{c}_mean"] = np.nanmean(v[:, :, j], axis=0)
            out[f"L_{c}_chg"] = v[-1, :, j] - v[0, :, j]
    return pd.DataFrame(out, index=t_ms)


def pnl(p: np.ndarray, r: np.ndarray, theta: float, fees: tuple[float, float], slip: float,
        sel: np.ndarray | None = None) -> dict:
    """確率が theta 以上（sel を渡せばその時刻）のとき t で買い t+h で売る。taker: 成行往復、maker: 手数料・滑りなし（楽観的な上限）。"""
    sel = p >= theta if sel is None else sel
    g = np.exp(r[sel])
    taker = g * (1 - slip) * (1 - fees[1]) / ((1 + slip) * (1 + fees[1])) - 1
    maker = g - 1
    def st(x):
        if len(x) < 3:
            return {"n": int(len(x))}
        return {"n": int(len(x)), "mean": float(x.mean()), "t": float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))),
                "win": float((x > 0).mean())}
    return {"theta": theta, "coverage": float(sel.mean()), "taker": st(taker), "maker_upper": st(maker)}


def _peak_gb() -> float:
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def run_direction(cfg: dict, spec: dict, workers: int | None = None) -> dict:
    market = load_market(cfg)
    workers = workers or max(1, os.cpu_count() or 1)
    start, end = to_utc(cfg["data"]["start"]), to_utc(cfg["data"]["end"])
    train_end = to_utc(spec["train_end"])
    bprm = barrier_params({**cfg, "pair_spec": market.spec}, market.spec)
    fees, slip = (bprm.maker_fee, bprm.taker_fee), bprm.s_slip
    h_min = int(spec.get("horizon_min", 15))
    h_ms = h_min * MIN_MS
    # 目的変数のしきい値: "taker_round_trip" なら成行往復の費用（手数料・滑りから計算）、数値ならその対数リターン、なければ 0
    tm = spec.get("target_min_return", 0.0)
    min_ret = taker_round_trip(fees[1], slip) if tm == "taker_round_trip" else float(tm)
    use_levels = bool(spec.get("level_summary", False))
    log.info("特徴量を計算 %s〜%s、ホライズン %d 分", start, end, h_min)
    F = feature_table(market, start, end, cfg["signal"]["sigma_span"], workers, h_ms, min_ret)
    del market  # 約定データは以降使わない（メモリを空ける）
    levels = F[PROFILE_COLUMNS] if use_levels else None  # 1 分ごとの生の水準（正解のない時刻も窓に使う）
    F = F[F["y"].notna()]
    log.info("特徴量 %d 行 × %d 列、最大メモリ %.1f GB", len(F), F.shape[1], _peak_gb())
    t_ms = F.index.to_numpy()
    feat_cols = [c for c in F.columns if c not in ("y", "r")]
    is_tr = t_ms < int(train_end.value // 1_000_000) - h_ms  # 正解が学習期間内で決まるものだけ
    is_te = t_ms >= int(train_end.value // 1_000_000)
    n_splits = int(spec.get("n_splits", 5))

    on_bar = (t_ms % h_ms) == 0
    a_tr, a_te = is_tr & on_bar, is_te & on_bar
    ta_tr, ta_te = t_ms[a_tr], t_ms[a_te]
    ra_te = F.loc[a_te, "r"].to_numpy()
    XA_tr, XA_te = F.loc[a_tr, feat_cols], F.loc[a_te, feat_cols]
    LS = (level_summary(levels, ta_tr, h_min), level_summary(levels, ta_te, h_min)) if use_levels else None

    def fit_side(y_all: np.ndarray, label: str) -> dict:
        """B（1 分ごと）と A の変種（A、A+B、A+B+L）を 1 つの正解で学習し、予測と指標を返す。"""
        yb_tr, yb_te = y_all[is_tr].astype(int), y_all[is_te].astype(int)
        log.info("モデル B（%s）: 学習 %d、評価 %d", label, int(is_tr.sum()), int(is_te.sum()))
        oof_b, test_b, folds_b, model_b = oof_and_final(F.loc[is_tr, feat_cols], yb_tr, t_ms[is_tr], F.loc[is_te, feat_cols],
                                               n_splits, h_ms)
        pred_b = pd.Series(np.concatenate([oof_b, test_b]), index=np.concatenate([t_ms[is_tr], t_ms[is_te]]))
        ya_tr, ya_te = y_all[a_tr].astype(int), y_all[a_te].astype(int)
        SB_tr, SB_te = stack_features(pred_b, ta_tr, h_min), stack_features(pred_b, ta_te, h_min)
        variants = {"A": (XA_tr, XA_te),
                    "A+B": (pd.concat([XA_tr, SB_tr.set_axis(XA_tr.index)], axis=1),
                            pd.concat([XA_te, SB_te.set_axis(XA_te.index)], axis=1))}
        if LS is not None:
            variants["A+B+L"] = (pd.concat([variants["A+B"][0], LS[0].set_axis(XA_tr.index)], axis=1),
                                 pd.concat([variants["A+B"][1], LS[1].set_axis(XA_te.index)], axis=1))
        log.info("モデル A（%s）: 学習 %d、評価 %d、変種 %s", label, len(XA_tr), len(XA_te), list(variants))
        oof, test, folds = {}, {}, {}
        for name, (Xtr, Xte) in variants.items():
            oof[name], test[name], folds[name], _ = oof_and_final(Xtr, ya_tr, ta_tr, Xte, n_splits, h_ms)
        test["B_on_bar"] = pred_b.reindex(ta_te).to_numpy()
        cv = {"B": {k: v for k, v in metrics(yb_tr, oof_b).items() if k != "calibration"}, "B_folds": folds_b}
        for name in variants:
            cv[name] = {k: v for k, v in metrics(ya_tr, oof[name]).items() if k != "calibration"}
            cv[f"{name}_folds"] = folds[name]
        te = {"B_all_minutes": metrics(yb_te, test_b), **{name: metrics(ya_te, test[name]) for name in test}}
        log.info("学習を終了（%s）、最大メモリ %.1f GB", label, _peak_gb())
        return {"ya_tr": ya_tr, "ya_te": ya_te, "variants": variants, "oof": oof, "test": test, "cv": cv, "metrics": te,
                "model_b": model_b}

    up = fit_side(F["y"].to_numpy(), "上げ")
    ya_te, variants, test = up["ya_te"], up["variants"], up["test"]
    two_sided = bool(spec.get("two_sided", False)) and min_ret > 0
    dn = None
    if two_sided:  # 下げ側: r < −c を 1 とする同じモデル（2026-09-26 オーナー指示）
        dn = fit_side(np.where(F["r"].to_numpy() < -min_ret, 1.0, 0.0), "下げ")

    thetas = list(spec.get("thetas", [0.55, 0.6]))
    top_fracs = list(spec.get("top_fracs", []))

    def pnl_rows(p, ths=None):
        rows = [pnl(p, ra_te, th, fees, slip) for th in (thetas if ths is None else ths)]
        for f in top_fracs:  # 評価期間の予測確率の上位 f（順位で選ぶ）。しきい値を評価期間で決めるので、説明用
            k = max(1, int(round(f * len(p))))
            top = np.zeros(len(p), bool)
            top[np.argsort(-p, kind="stable")[:k]] = True
            rows.append({**pnl(p, ra_te, float(p[top].min()), fees, slip, sel=top), "top_frac": f})
        return rows
    res = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "horizon_min": h_min,
        "periods": {"train": [str(start), str(train_end)], "test": [str(train_end), str(end)]},
        "n": {"B_train": int(is_tr.sum()), "B_test": int(is_te.sum()), "A_train": int(len(XA_tr)), "A_test": int(len(XA_te))},
        "base_rate_test": float(ya_te.mean()) if len(ya_te) else None,
        "features": feat_cols,
        "variant_features": {name: list(Xtr.columns) for name, (Xtr, _) in variants.items()},
        "cv": up["cv"],
        "test": up["metrics"],
        "target_min_return": min_ret,
        "pnl_test": {name: pnl_rows(p) for name, p in test.items()},
        "unconditional_test": pnl(np.ones(len(ra_te)), ra_te, 0.0, fees, slip),
        "fees": {"maker": fees[0], "taker": fees[1], "s_slip": slip},
    }
    if dn is not None:
        # 信号 = P(上げ > c) − P(下げ > c)。同じ変種どうしの差
        diff = {name: test[name] - dn["test"][name] for name in test}
        diff_thetas = list(spec.get("diff_thetas", [0.05, 0.1, 0.2]))
        res["two_sided"] = {
            "base_rate_down_test": float(dn["ya_te"].mean()) if len(dn["ya_te"]) else None,
            "cv_down": dn["cv"], "test_down": dn["metrics"],
            "corr_up_down_test": {name: float(np.corrcoef(test[name], dn["test"][name])[0, 1]) for name in test},
            "diff_stats": {name: {"mean": float(d.mean()), "std": float(d.std()), "p95": float(np.percentile(d, 95))}
                           for name, d in diff.items()},
            # 差が正の側（買い）。差が小さい側は空売りできないので、値動きの平均だけ参考に出す
            "pnl_diff": {name: pnl_rows(d, diff_thetas) for name, d in diff.items()},
            "bottom_mean_r": {name: {str(f): float(np.mean(ra_te[np.argsort(d, kind="stable")[:max(1, int(round(f * len(d))))]]))
                                     for f in top_fracs} for name, d in diff.items()},
        }
    # AUC の差（各変種 − A）の日単位ブロック・ブートストラップ
    days = (ta_te // (24 * H_MS)).astype(np.int64)
    uniq = np.unique(days)
    rng = np.random.default_rng(0)
    from sklearn.metrics import roc_auc_score

    idx_by_day = {d: np.flatnonzero(days == d) for d in uniq}
    diffs = {name: [] for name in variants if name != "A"}
    for _ in range(int(spec.get("n_boot", 200))):
        pick = np.concatenate([idx_by_day[d] for d in rng.choice(uniq, size=len(uniq), replace=True)])
        yy = ya_te[pick]
        if len(np.unique(yy)) < 2:
            continue
        base = roc_auc_score(yy, test["A"][pick])
        for name in diffs:
            diffs[name].append(roc_auc_score(yy, test[name][pick]) - base)
    res["auc_diff_vs_A"] = {name: {"mean": float(np.mean(d)), "p05": float(np.percentile(d, 5)),
                                   "p95": float(np.percentile(d, 95))} for name, d in diffs.items() if d}
    # 実験ログ（試行として数える）
    th = params_hash({"direction": spec, "features": feat_cols, "fees": list(fees), "s_slip": slip})
    log_path = Path(cfg.get("experiment_log", "reports/experiments.jsonl"))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        for name in test:
            fh.write(json.dumps({"run_at": res["run_at"], "stage": "direction", "trial_hash": th,
                                 "code_version": os.environ.get("GITHUB_SHA"),
                                 "key": f"direction/{h_min}m/r>{min_ret:.4f}/{'two_sided/' if dn else ''}{name}",
                                 "test_auc": res["test"][name].get("auc")}) + "\n")
    res["trial_hash"] = th
    if spec.get("export_model"):
        # 本番（dashboard/app）で使う B（上げ側、1 分ごと）。木は JSON に書き出し、純粋 Python（treeeval）で評価する
        res["export"] = {
            "model": up["model_b"].booster_.dump_model(),
            "features": feat_cols, "horizon_min": h_min, "target_min_return": min_ret,
            "sigma_span": int(cfg["signal"]["sigma_span"]), "train_end": str(train_end),
            "large_trade_amount": cfg["data"].get("large_trade_amount"),
            "test_auc": res["test"]["B_all_minutes"].get("auc"), "base_rate_test": res["base_rate_test"],
            "trial_hash": th, "run_at": res["run_at"],
        }
    return res


def _p(x, pct=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x * 100:+.3f}%" if pct else f"{x:.4f}"


def render(rep: dict) -> str:
    h = rep.get("horizon_min", 15)
    mr = rep.get("target_min_return", 0.0)
    what = f"{h} 分後に {mr * 100:.2f}% を超えて上がるか" if mr > 0 else f"{h} 分後の上げ下げ"
    md = [f"# {what}の予測（二段構え、{rep['run_at']}）\n",
          f"- 学習・交差検証: {rep['periods']['train'][0]} 〜 {rep['periods']['train'][1]}",
          f"- 評価（1 回だけ）: {rep['periods']['test'][0]} 〜 {rep['periods']['test'][1]}（これまでの分析で一部を見ている期間）",
          f"- 標本: B 学習 {rep['n']['B_train']:,} / 評価 {rep['n']['B_test']:,}、A 学習 {rep['n']['A_train']:,} / 評価 {rep['n']['A_test']:,}",
          f"- 目的変数: {what}（1 = はい）。評価期間で 1 の割合: {_p(rep['base_rate_test'])[1:]}、特徴量 {len(rep['features'])} 個"
          + (f"（A+B+L は {len(rep['variant_features']['A+B+L'])} 個）" if "A+B+L" in rep.get("variant_features", {}) else "") + "\n",
          "## 当たり具合（AUC、0.5 が当て推量）\n",
          "| モデル | 交差検証（学習期間） | 評価期間 | 評価期間 log loss |\n|---|---|---|---|"]
    cv, te = rep["cv"], rep["test"]
    rows = [("B（1 分ごと、全時刻）", "B", "B_all_minutes"), (f"B（{h} 分の区切りだけ）", None, "B_on_bar"),
            (f"A（{h} 分ごと）", "A", "A"), ("A+B（二段構え: A + 直近の B の要約）", "A+B", "A+B")]
    if "A+B+L" in te:
        rows.append(("A+B+L（さらに直近の TPO・価格帯別出来高の水準の推移）", "A+B+L", "A+B+L"))
    for name, cvk, tek in rows:
        c = cv.get(cvk, {}) if cvk else {}
        md.append(f"| {name} | {_p(c.get('auc'), False)} | {_p(te[tek].get('auc'), False)} | {_p(te[tek].get('log_loss'), False)} |")
    for name, d in rep.get("auc_diff_vs_A", {}).items():
        md.append(f"\n{name} と A の AUC の差（評価期間、日単位ブートストラップ）: 平均 {_p(d['mean'], False)}、"
                  f"5〜95% 区間 [{_p(d['p05'], False)}, {_p(d['p95'], False)}]")
    md.append(f"\n## 費用込みの損益（評価期間、確率がしきい値以上で買い {h} 分後に売る）\n")
    md.append("| モデル | しきい値 | 発注割合 | 取引 | 成行往復の平均 | t 値 | 勝率 | 手数料・滑りなしの平均（上限） |\n|---|---|---|---|---|---|---|---|")
    for name, rows in rep["pnl_test"].items():
        for r in rows:
            tk, mk = r["taker"], r["maker_upper"]
            th = f"上位 {r['top_frac'] * 100:.0f}%（{r['theta']:.3f}）" if "top_frac" in r else f"{r['theta']}"
            md.append(f"| {name} | {th} | {_p(r['coverage'])[1:]} | {tk.get('n', 0)} | {_p(tk.get('mean'))} | "
                      f"{_p(tk.get('t'), False)} | {_p(tk.get('win'))[1:]} | {_p(mk.get('mean'))} |")
    u = rep["unconditional_test"]
    md.append(f"| 常に買う | - | 100% | {u['taker'].get('n', 0)} | {_p(u['taker'].get('mean'))} | "
              f"{_p(u['taker'].get('t'), False)} | {_p(u['taker'].get('win'))[1:]} | {_p(u['maker_upper'].get('mean'))} |")
    if "two_sided" in rep:
        ts = rep["two_sided"]
        md.append(f"\n## 下げ側（{h} 分後に {mr * 100:.2f}% を超えて下がるか）と、上げ − 下げの信号\n")
        md.append(f"評価期間で下げ側の 1 の割合: {_p(ts['base_rate_down_test'])[1:]}\n")
        md.append("| モデル | 上げ側 AUC | 下げ側 AUC | 上げ確率と下げ確率の相関 | 差の平均 | 差の標準偏差 |\n|---|---|---|---|---|---|")
        for name in rep["test"]:
            if name == "B_all_minutes":
                md.append(f"| B（1 分ごと、全時刻） | {_p(rep['test'][name].get('auc'), False)} | {_p(ts['test_down'][name].get('auc'), False)} | - | - | - |")
                continue
            c, d = ts["corr_up_down_test"][name], ts["diff_stats"][name]
            md.append(f"| {name} | {_p(rep['test'][name].get('auc'), False)} | {_p(ts['test_down'][name].get('auc'), False)} | "
                      f"{_p(c, False)} | {_p(d['mean'], False)} | {_p(d['std'], False)} |")
        md.append(f"\n### 信号（上げ確率 − 下げ確率）がしきい値以上で買い {h} 分後に売る\n")
        md.append("| モデル | しきい値 | 発注割合 | 取引 | 成行往復の平均 | t 値 | 勝率 | 手数料・滑りなしの平均（上限） |\n|---|---|---|---|---|---|---|---|")
        for name, rows in ts["pnl_diff"].items():
            for r in rows:
                tk, mk = r["taker"], r["maker_upper"]
                th = f"上位 {r['top_frac'] * 100:.0f}%（{r['theta']:.3f}）" if "top_frac" in r else f"{r['theta']}"
                md.append(f"| {name} | {th} | {_p(r['coverage'])[1:]} | {tk.get('n', 0)} | {_p(tk.get('mean'))} | "
                          f"{_p(tk.get('t'), False)} | {_p(tk.get('win'))[1:]} | {_p(mk.get('mean'))} |")
        md.append("\n参考: 信号が最も低い側（空売りはできないので取引しない）の値動きの平均（手数料なし）")
        for name, d in ts["bottom_mean_r"].items():
            md.append(f"- {name}: " + "、".join(f"下位 {float(f) * 100:.0f}% {_p(v)}" for f, v in d.items()))
    md.append(f"\n手数料: メイカー {rep['fees']['maker']}、テイカー {rep['fees']['taker']}、成行の滑り {rep['fees']['s_slip']}")
    return "\n".join(md) + "\n"


def main(args) -> None:
    from .pipeline import load_config

    rep = run_direction(load_config(args.config), yaml.safe_load(Path(args.spec).read_text()), workers=args.workers)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    export = rep.pop("export", None)
    if export:
        (out / "model_B.json").write_text(json.dumps(export, separators=(",", ":")))
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    md = render(rep)
    (out / "report.md").write_text(md)
    print(md)
