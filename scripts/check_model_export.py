"""書き出したモデル（model_B.json）を純粋 Python の評価器で読めることと、木の数・特徴量の数を確認する。"""
import json
import math
import sys

from bbresearch.treeeval import predict_proba

path = sys.argv[1]
m = json.load(open(path))
feats = m["features"]
trees = m["model"]["tree_info"]
assert m["model"]["feature_names"] == feats, "特徴量の順序がモデルと一致しない"
p0 = predict_proba(m["model"], [0.0] * len(feats))
p1 = predict_proba(m["model"], [float("nan")] * len(feats))
assert 0.0 < p0 < 1.0 and 0.0 < p1 < 1.0
print(f"ok: {len(trees)} 本の木、{len(feats)} 個の特徴量、ホライズン {m['horizon_min']} 分、"
      f"しきい値 {m['target_min_return']:.4f}、評価期間の AUC {m['test_auc']:.4f}、"
      f"{sum(len(json.dumps(t)) for t in trees) / 1e6:.1f} MB")
