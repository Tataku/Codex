import datetime as dt
import random

from mc_core import (
    DayFeature,
    Observation,
    SimulationConfig,
    build_return_samples,
    compute_ma_features,
    evaluate_at_index,
    choose_pool,
    simulate_paths,
)


def _sample_obs(n=40):
    base = dt.date(2024, 1, 1)
    return [Observation(date=base + dt.timedelta(days=i), close=100 + i, divergence_pct=float(i % 7)) for i in range(n)]


def test_no_lookahead_ma_matches_truncated_series():
    obs = _sample_obs(20)
    full = compute_ma_features(obs, ma_window=5)
    t = 12
    truncated = compute_ma_features(obs[: t + 1], ma_window=5)
    assert full[t].ma_value == truncated[-1].ma_value


def test_seed_determinism():
    obs = _sample_obs(45)
    features = compute_ma_features(obs, ma_window=5)
    samples = build_return_samples(features)
    pool, _ = choose_pool(samples, current_divergence=2.0, current_regime="bull", bandwidth=10.0, min_bucket_samples=1, ma_filter_mode="both")
    cfg = SimulationConfig(paths=30, horizon_days=5, divergence_bandwidth=10.0, min_bucket_samples=1, seed=123)

    p1 = simulate_paths(features[-1].obs.close, pool, cfg, random.Random(123))
    p2 = simulate_paths(features[-1].obs.close, pool, cfg, random.Random(123))
    assert p1 == p2


def test_backtest_forward_return_alignment_formula():
    obs = _sample_obs(10)
    features = [DayFeature(obs=o, ma_value=100.0, ma_regime="bull") for o in obs]
    i = 2
    h = 3
    realized = (features[i + h].obs.close / features[i].obs.close) - 1.0
    expected = (105 / 102) - 1.0
    assert abs(realized - expected) < 1e-12


def test_simulate_backtest_state_alignment():
    obs = _sample_obs(60)
    features = compute_ma_features(obs, ma_window=10)
    cfg = SimulationConfig(paths=20, horizon_days=7, divergence_bandwidth=5.0, min_bucket_samples=1, seed=77)
    idx = 40

    a = evaluate_at_index(features, idx, cfg, random.Random(77), "both", 0)
    b = evaluate_at_index(features, idx, cfg, random.Random(77), "both", 0)
    assert a is not None and b is not None
    assert a["anchor"].obs.date == b["anchor"].obs.date
    assert a["fallback_used"] == b["fallback_used"]
    assert len(a["pool"]) == len(b["pool"])
