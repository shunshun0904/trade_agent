import numpy as np
import pandas as pd
import pytest

from bbresearch.treeeval import predict_proba, predict_rows

lgb = pytest.importorskip("lightgbm")


def _fit(X, y, **kw):
    return lgb.LGBMClassifier(n_estimators=40, num_leaves=8, learning_rate=0.1, min_child_samples=5, verbose=-1,
                              random_state=0, **kw).fit(X, y)


def test_matches_lightgbm_including_nan():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 7))
    y = ((X[:, 0] * X[:, 1] + X[:, 2] > 0.2) | (X[:, 3] > 1.5)).astype(int)
    X[rng.random(X.shape) < 0.1] = np.nan  # 学習時から欠損あり（missing_type が NaN になる）
    X[:, 6] = 0.0  # 使われない列
    m = _fit(X, y)
    model = m.booster_.dump_model()
    p_ref = m.predict_proba(X)[:, 1]
    p = predict_rows(model, X.tolist())
    assert np.allclose(p, p_ref, atol=1e-9)
    assert {"NaN", "None"} & {n for t in model["tree_info"] for n in _missing_types(t["tree_structure"])}


def test_matches_lightgbm_when_trained_without_nan_and_scored_with_nan():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
    m = _fit(X, y)
    model = m.booster_.dump_model()
    Xs = X[:200].copy()
    Xs[::3, 0] = np.nan  # 学習時になかった欠損: LightGBM は 0 として扱う
    assert np.allclose(predict_rows(model, Xs.tolist()), m.predict_proba(Xs)[:, 1], atol=1e-9)


def test_feature_names_from_dataframe_and_none_as_missing():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(1500, 3)), columns=["a", "b", "c"])
    df.loc[df.index % 5 == 0, "b"] = np.nan
    y = (df["a"] - df["b"].fillna(0) > 0).astype(int)
    m = _fit(df, y)
    model = m.booster_.dump_model()
    assert model["feature_names"] == ["a", "b", "c"]
    row = df.iloc[0].tolist()
    assert abs(predict_proba(model, row) - m.predict_proba(df.iloc[[0]])[0, 1]) < 1e-9
    row_none = [row[0], None, row[2]]
    row_nan = [row[0], float("nan"), row[2]]
    assert predict_proba(model, row_none) == predict_proba(model, row_nan)


def _missing_types(node):
    if "leaf_value" in node:
        return []
    return [node["missing_type"]] + _missing_types(node["left_child"]) + _missing_types(node["right_child"])
