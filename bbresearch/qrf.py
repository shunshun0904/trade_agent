"""分位点回帰フォレスト（Meinshausen 2006）と、予測分布の評価。2026-09-27 オーナー決定。

ランダムフォレストを普通に（平均を当てるように）学習し、予測時には「テスト点と同じ葉に落ちた学習点」に重みを配って
学習時の目的変数の重み付き経験分布を条件付き分布とする。分布の形は仮定しない。

重み w_i(x) = (1/T) Σ_t 1[x と x_i が木 t で同じ葉] / (その葉の学習点の数)。

評価:
- ピンボール損失（分位点ごと）、CRPS（重み付き経験分布から閉じた式で）、較正（予測分位点を下回った割合）。
- 比較対象: 無条件の経験分布（学習期間全体）と、直近 n 本の経験分布（ボラの変化だけを追う基準）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestRegressor

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class QuantileForest:
    def __init__(self, n_estimators: int = 300, min_samples_leaf: int = 50, max_features: float | str = 0.33,
                 max_samples: float | None = 0.5, random_state: int = 0, n_jobs: int = -1,
                 split_target: str = "mean", n_bins: int = 8):
        """split_target: "mean" は普通の回帰木（平均の差で分割）。"bins" は目的変数を分位点で n_bins 個の区間に分け、
        その one-hot を多出力にして学習する（区間の割合 = 分布の形が違う領域を分ける。GRF の分位点フォレストの分割規則の近似。
        2026-09-27 オーナー決定）。葉の分布は元の y から作るので、予測の仕組みは同じ。"""
        self.rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                        max_features=max_features, max_samples=max_samples, bootstrap=True,
                                        random_state=random_state, n_jobs=n_jobs)
        self.split_target, self.n_bins = split_target, n_bins
        self._y: np.ndarray | None = None
        self._order: np.ndarray | None = None
        self._leaf_mats: list[sparse.csr_matrix] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "QuantileForest":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if self.split_target == "bins":
            edges = np.quantile(y, np.linspace(0, 1, self.n_bins + 1)[1:-1])
            target = np.eye(self.n_bins)[np.digitize(y, edges)]
        else:
            target = y
        self.rf.fit(X, target)
        self._order = np.argsort(y, kind="stable")
        self._y = y[self._order]
        leaves = self.rf.apply(X)  # (n_train, n_trees)
        n = len(y)
        self._leaf_mats = []
        for t in range(leaves.shape[1]):
            leaf = leaves[self._order, t]
            counts = np.bincount(leaf)
            # 行 = 葉、列 = 学習点（y の昇順）。値 = 1 / 葉の学習点の数
            m = sparse.csr_matrix((1.0 / counts[leaf], (leaf, np.arange(n))), shape=(counts.size, n))
            self._leaf_mats.append(m)
        return self

    def weights(self, X: np.ndarray) -> np.ndarray:
        """各テスト点の学習点への重み（y の昇順に並んだ列）。形 (n_test, n_train)。"""
        X = np.asarray(X, dtype=float)
        leaves = self.rf.apply(X)
        n_test, T = leaves.shape
        # 木ごとに「テスト点が落ちた葉の行」を取り出して縦に積み、選択行列で木の和をとる（疎行列の積 1 回）
        stacked = sparse.vstack([m[leaves[:, t]] for t, m in enumerate(self._leaf_mats)], format="csr")
        sel = sparse.csr_matrix((np.ones(T * n_test), (np.tile(np.arange(n_test), T), np.arange(T * n_test))),
                                shape=(n_test, T * n_test))
        return (sel @ stacked).toarray() / T

    def predict_quantiles(self, X: np.ndarray, qs=QUANTILES, chunk: int = 512) -> np.ndarray:
        """条件付き分位点。形 (n_test, len(qs))。テスト点をまとめて処理し、メモリを抑える。"""
        X = np.asarray(X, dtype=float)
        out = np.empty((len(X), len(qs)))
        for s in range(0, len(X), chunk):
            w = self.weights(X[s:s + chunk])
            out[s:s + chunk] = weighted_quantiles(self._y, w, qs)
        return out

    def predict_distribution(self, X: np.ndarray, chunk: int = 512):
        """(sorted_y, weights) を chunk ごとに返すジェネレータ。CRPS や P(r > c) の計算に使う。"""
        X = np.asarray(X, dtype=float)
        for s in range(0, len(X), chunk):
            yield self._y, self.weights(X[s:s + chunk])


def weighted_quantiles(y_sorted: np.ndarray, w: np.ndarray, qs) -> np.ndarray:
    """y_sorted は昇順、w は (n_test, n) の重み（行和 1）。各行の分位点を返す。"""
    cw = np.cumsum(w, axis=1)
    cw /= cw[:, -1:]
    out = np.empty((w.shape[0], len(qs)))
    for j, q in enumerate(qs):
        idx = np.minimum((cw < q).sum(axis=1), len(y_sorted) - 1)
        out[:, j] = y_sorted[idx]
    return out


def crps_weighted(y_sorted: np.ndarray, w: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """重み付き経験分布の CRPS: E|Y − obs| − ½ E|Y − Y'|（各行）。"""
    w = w / w.sum(axis=1, keepdims=True)
    term1 = (w * np.abs(y_sorted[None, :] - obs[:, None])).sum(axis=1)
    cw = np.cumsum(w, axis=1)
    before = cw - w                       # 自分より前の重みの和
    after = 1.0 - cw                      # 自分より後の重みの和
    e_abs = 2.0 * (w * y_sorted[None, :] * (before - after)).sum(axis=1)
    return term1 - 0.5 * e_abs


def prob_exceed(y_sorted: np.ndarray, w: np.ndarray, c: float) -> np.ndarray:
    """P(Y > c)（各行）。"""
    w = w / w.sum(axis=1, keepdims=True)
    return w[:, y_sorted > c].sum(axis=1)


def prob_exceed_rows(y_sorted: np.ndarray, w: np.ndarray, c: np.ndarray) -> np.ndarray:
    """P(Y > c_i)（行ごとに違うしきい値）。"""
    w = w / w.sum(axis=1, keepdims=True)
    cw = np.cumsum(w, axis=1)
    idx = np.searchsorted(y_sorted, np.asarray(c, dtype=float), side="right")   # y_sorted[:idx] <= c
    below = np.where(idx > 0, cw[np.arange(len(c)), np.maximum(idx - 1, 0)], 0.0)
    return 1.0 - below


def pinball(obs: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    d = obs - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def empirical_quantiles(y: np.ndarray, qs=QUANTILES) -> np.ndarray:
    return np.quantile(y, qs)


def rolling_empirical(y_train_tail: np.ndarray, y_test: np.ndarray, n: int, horizon: int, qs=QUANTILES) -> np.ndarray:
    """時点 i の直近 n 本の「確定した」収益率（i − horizon 以前）の分位点。ボラの変化を追う基準。

    y_train_tail は学習期間の末尾（少なくとも n + horizon 本）、y_test は評価期間の目的変数（時系列順）。
    """
    y_all = np.concatenate([y_train_tail, y_test])
    start = len(y_train_tail)
    out = np.empty((len(y_test), len(qs)))
    for i in range(len(y_test)):
        end = start + i - horizon              # 行 i の時点で確定している最後の収益率は i − horizon
        win = y_all[max(0, end - n):end]
        out[i] = np.quantile(win, qs) if len(win) >= 20 else np.nan
    return out


def evaluate_quantiles(obs: np.ndarray, q_pred: np.ndarray, qs=QUANTILES) -> dict:
    ok = ~np.isnan(q_pred).any(axis=1)
    obs, q_pred = obs[ok], q_pred[ok]
    res = {"n": int(ok.sum()), "pinball": {}, "coverage": {}}
    for j, q in enumerate(qs):
        res["pinball"][q] = pinball(obs, q_pred[:, j], q)
        res["coverage"][q] = float(np.mean(obs <= q_pred[:, j]))
    res["pinball_mean"] = float(np.mean(list(res["pinball"].values())))
    lo, hi = qs.index(0.05), qs.index(0.95)
    res["interval90"] = float(np.mean((obs >= q_pred[:, lo]) & (obs <= q_pred[:, hi])))
    res["width90_median"] = float(np.median(q_pred[:, hi] - q_pred[:, lo]))
    return res


def reliability(p: np.ndarray, hit: np.ndarray, bins=(0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01)) -> pd.DataFrame:
    """予測確率 p の区間ごとに、実際に起きた割合と件数。"""
    idx = np.digitize(p, bins) - 1
    rows = []
    for b in range(len(bins) - 1):
        m = idx == b
        rows.append({"bin": f"{bins[b]:.1f}-{min(bins[b + 1], 1.0):.1f}", "n": int(m.sum()),
                     "p_mean": float(p[m].mean()) if m.any() else np.nan, "hit_rate": float(hit[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)
