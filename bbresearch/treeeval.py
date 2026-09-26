"""LightGBM の二値分類モデル（Booster.dump_model() の JSON）を標準ライブラリだけで評価する。

本番（dashboard/app）で lightgbm を同梱しないために使う。研究用の predict_proba と一致することを
tests/test_treeeval.py で照合する。欠損値（None / NaN）の扱いは LightGBM と同じ:
- missing_type が "NaN": NaN は default_left に従う
- missing_type が "Zero": NaN は 0 として比べる（欠損は 0 扱い）
- missing_type が "None": 欠損はない前提。NaN は 0 として比べる（LightGBM の実装と同じ）
"""
from __future__ import annotations

import math


def _is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _leaf_value(node: dict, x: list) -> float:
    while "leaf_value" not in node:
        v = x[node["split_feature"]]
        mt = node.get("missing_type", "None")
        if _is_missing(v):
            if mt == "NaN":
                go_left = node["default_left"]
            else:  # "Zero" と "None": 0 として比べる
                v = 0.0
                go_left = _compare(node, v)
        else:
            go_left = _compare(node, v)
        node = node["left_child"] if go_left else node["right_child"]
    return node["leaf_value"]


def _compare(node: dict, v: float) -> bool:
    dt = node["decision_type"]
    th = node["threshold"]
    if dt == "<=":
        return v <= th
    if dt == "<":
        return v < th
    if dt == "==":  # カテゴリ分割（このプロジェクトでは使わない）
        return any(v == float(c) for c in str(th).split("||"))
    raise ValueError(f"decision_type {dt} は扱えない")


def raw_score(model: dict, x: list) -> float:
    return sum(_leaf_value(t["tree_structure"], x) for t in model["tree_info"])


def predict_proba(model: dict, x: list) -> float:
    """1 行の特徴量（model["feature_names"] の順）に対する正例の確率。"""
    if model.get("num_class", 1) != 1 or not str(model.get("objective", "")).startswith("binary"):
        raise ValueError("二値分類（binary）のモデルだけを扱う")
    s = raw_score(model, x)
    if s >= 0:
        return 1.0 / (1.0 + math.exp(-s))
    e = math.exp(s)
    return e / (1.0 + e)


def predict_rows(model: dict, rows: list[list]) -> list[float]:
    return [predict_proba(model, r) for r in rows]
