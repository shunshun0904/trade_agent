"""bbresearch/fcompare.py: 損失差の検定（fixed-b、EWC、ブートストラップ、ホルム法）。"""
import math

import numpy as np

from bbresearch import fcompare as fc


def test_nw_lrv_matches_explicit_bartlett_sum():
    rng = np.random.default_rng(0)
    d = rng.standard_normal(300)
    M = 12
    e = d - d.mean()
    g = [np.sum(e[j:] * e[:len(e) - j]) / len(e) for j in range(M)]
    expected = g[0] + 2 * sum((1 - j / M) * g[j] for j in range(1, M))
    assert math.isclose(fc.nw_lrv(d, M), expected, rel_tol=1e-9)


def test_fixedb_null_quantile_matches_kiefer_vogelsang_polynomial():
    P = 500
    M = fc.nw_bandwidth(P)            # ⌈1.3 √500⌉ = 30、b = 0.06
    assert M == 30
    q = np.quantile(fc.fixedb_null(P, M, 10000, 0), 0.975)
    assert abs(q - fc.fixedb_cv975(M / P)) < 0.06
    # p 値は |t| が大きいほど小さい
    assert fc.fixedb_pvalue(3.0, P, M) < fc.fixedb_pvalue(1.0, P, M)


def test_ewc_uses_cosine_projections():
    rng = np.random.default_rng(1)
    d = rng.standard_normal(400) + 0.1
    P, B = len(d), fc.ewc_dof(len(d))
    t_idx = np.arange(1, P + 1)
    lam = np.array([math.sqrt(2 / P) * np.sum(np.cos(math.pi * j * (t_idx - 0.5) / P) * d) for j in range(1, B + 1)])
    t_expected = math.sqrt(P) * d.mean() / math.sqrt(np.mean(lam**2))
    t, b = fc.ewc_t(d)
    assert b == B == int(0.4 * 400 ** (2 / 3))
    assert math.isclose(t, t_expected, rel_tol=1e-9)


def test_tests_have_roughly_correct_size_under_iid_null():
    rng = np.random.default_rng(2)
    rej_fb = rej_ewc = 0
    R = 200
    for _ in range(R):
        c = fc.compare(rng.standard_normal(400), fixedb_sims=2000)
        rej_fb += c["p_fb"] < 0.05
        rej_ewc += c["p_ewc"] < 0.05
    assert rej_fb / R < 0.1 and rej_ewc / R < 0.1


def test_compare_detects_a_clear_mean_difference_and_bootstrap_runs():
    rng = np.random.default_rng(3)
    c = fc.compare(rng.standard_normal(2000) - 0.2, fixedb_sims=2000, bootstrap_reps=199, adf=True)
    assert c["mean"] < 0 and c["p_fb"] < 0.01 and c["p_ewc"] < 0.01 and c["p_sb"] < 0.05
    assert c["adf_p"] < 0.05 and set(c["acf"]) == {1, 4, 24, 168}


def test_holm_adjustment():
    assert np.allclose(fc.holm([0.01, 0.04, 0.03, 0.2]), [0.04, 0.09, 0.09, 0.2])
