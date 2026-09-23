"""Phase 5: メタモデル（学習・交差検証・保存）。

学習データは約定したイベントのみ。dip と breakout は別々に学習する（D6）。
- logit: ロジスティック回帰（ベースライン）。欠損は学習 fold の中央値で埋め、標準化も学習 fold だけで推定する
- lgbm:  LightGBM（欠損はそのまま扱う）
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cv import PurgedKFold


def make_model(kind: str, seed: int = 0, params: dict | None = None):
    params = dict(params or {})
    if kind == "logit":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=params.pop("C", 1.0), max_iter=2000),
        )
    if kind == "lgbm":
        import lightgbm as lgb

        base = dict(
            n_estimators=200, learning_rate=0.03, num_leaves=15, min_child_samples=50,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, verbose=-1, n_jobs=1,
        )
        base.update(params)
        return lgb.LGBMClassifier(**base)
    raise ValueError(f"未知のモデル: {kind}")


def _fit(model, X: pd.DataFrame, y: np.ndarray, w: np.ndarray | None):
    if hasattr(model, "steps"):
        return model.fit(X, y, **({"logisticregression__sample_weight": w} if w is not None else {}))
    return model.fit(X, y, sample_weight=w)


def _const_proba(y: np.ndarray, w: np.ndarray | None, n: int) -> np.ndarray:
    p = float(np.average(y, weights=w)) if len(y) else 0.5
    return np.full(n, p)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def metrics(y: np.ndarray, p: np.ndarray, thresholds=(0.5, 0.55, 0.6)) -> dict:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out: dict = {"n": int(len(y)), "base_rate": float(y.mean()) if len(y) else None}
    if len(np.unique(y)) == 2:
        out["log_loss"] = float(log_loss(y, p))
        out["auc"] = float(roc_auc_score(y, p))
        out["brier"] = float(brier_score_loss(y, p))
    for th in thresholds:
        sel = p >= th
        tp = int((sel & (y == 1)).sum())
        out[f"precision@{th}"] = tp / int(sel.sum()) if sel.any() else None
        out[f"recall@{th}"] = tp / int((y == 1).sum()) if (y == 1).any() else None
        out[f"coverage@{th}"] = float(sel.mean())
    # 較正: 予測確率の十分位ごとの平均予測と実現率
    if len(y) >= 20:
        bins = pd.qcut(p, q=min(10, len(np.unique(p))), duplicates="drop")
        cal = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
        out["calibration"] = [{"p_mean": float(a), "y_rate": float(b)} for a, b in zip(cal["p"], cal["y"])]
    return out


@dataclass
class CVResult:
    oof: pd.Series                       # 学習データ上の out-of-fold 予測確率
    fold_metrics: list[dict] = field(default_factory=list)
    overall: dict = field(default_factory=dict)
    models: list = field(default_factory=list)          # fold ごとのモデル（学習できなかった fold は None）
    test_starts: list = field(default_factory=list)     # fold ごとのテスト区間の最初の t0
    fallback: float = 0.5                                # モデルがない fold の予測値（学習データの陽性率）


def cross_validate(
    kind: str, X: pd.DataFrame, y: pd.Series, w: pd.Series | None, t0: pd.Series, t_x: pd.Series,
    n_splits: int, embargo: pd.Timedelta, seed: int = 0, params: dict | None = None,
) -> CVResult:
    """Purged K-fold で out-of-fold 予測を作る。X, y, w, t0, t_x は t0 昇順で同じ index。"""
    oof = pd.Series(np.nan, index=X.index)
    folds, models, starts = [], [], []
    cv = PurgedKFold(n_splits=n_splits, embargo=embargo)
    yv = y.to_numpy(dtype=int)
    wv = None if w is None else w.to_numpy(dtype=float)
    for k, (tr, te) in enumerate(cv.split(t0, t_x)):
        starts.append(pd.Timestamp(t0.iloc[te[0]]) if hasattr(t0, "iloc") else pd.Timestamp(t0[te[0]]))
        if len(np.unique(yv[tr])) < 2:
            m = None
            p = _const_proba(yv[tr], None if wv is None else wv[tr], len(te))
        else:
            m = _fit(make_model(kind, seed, params), X.iloc[tr], yv[tr], None if wv is None else wv[tr])
            p = predict_proba(m, X.iloc[te])
        models.append(m)
        oof.iloc[te] = p
        folds.append({"fold": k, "n_train": int(len(tr)), "n_test": int(len(te)), **metrics(yv[te], p)})
    return CVResult(oof=oof, fold_metrics=folds, overall=metrics(yv, oof.to_numpy()), models=models,
                    test_starts=starts, fallback=float(yv.mean()))


def predict_by_fold(res: CVResult, X: pd.DataFrame, t0: pd.Series) -> pd.Series:
    """t0 を含むテスト区間の fold モデルで予測する。

    ラベルのないイベント（未約定）にも発注判断のための確率が必要なので、そのイベントを
    テスト側に含む fold のモデルを使う（そのイベント付近のラベルで学習していないモデル）。
    """
    out = pd.Series(np.nan, index=X.index)
    if X.empty or not res.models:
        return out
    starts = pd.DatetimeIndex(res.test_starts)
    k = np.clip(starts.searchsorted(pd.DatetimeIndex(t0), side="right") - 1, 0, len(res.models) - 1)
    for j, m in enumerate(res.models):
        sel = k == j
        if sel.any():
            out[sel] = predict_proba(m, X[sel]) if m is not None else res.fallback
    return out


def fit_final(kind: str, X: pd.DataFrame, y: pd.Series, w: pd.Series | None, seed: int = 0, params: dict | None = None):
    return _fit(make_model(kind, seed, params), X, y.to_numpy(dtype=int), None if w is None else w.to_numpy(dtype=float))


def params_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def save_model(model, path: Path, meta: dict) -> None:
    """モデルと再現用メタデータ（特徴量一覧、パラメータのハッシュ、学習期間など）を保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path.with_suffix(".joblib"))
    path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
