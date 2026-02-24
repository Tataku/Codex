#!/usr/bin/env python3
"""BTC Monte Carlo projections conditioned on divergence + MA regime, with backtesting."""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from sim.mc_core import SimulationConfig, compute_ma_features, load_observations, resolve_as_of_idx

# local bool parser to keep CLI behavior explicit
def parse_bool(raw: str) -> bool:
    normalized = str(raw).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {raw}")

from sim.mc_core import (
    run_backtest_collect,
    simulate_distribution_batches,
    write_csv,
    write_json,
)


def _parse_date(raw: str | None) -> dt.date | None:
    return dt.date.fromisoformat(raw) if raw else None


def _build_sim_config(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        paths=args.paths,
        horizon_days=args.horizon_days,
        divergence_bandwidth=args.divergence_bandwidth,
        min_bucket_samples=args.min_bucket_samples,
        seed=args.seed,
    )


def run_simulate(args: argparse.Namespace) -> None:
    observations = load_observations(args.xlsx)
    features = compute_ma_features(observations, args.ma_window)
    if len(features) < 400:
        raise SystemExit("Not enough observations")

    as_of = _parse_date(args.as_of_date) or observations[-1].date
    as_of_idx = resolve_as_of_idx(observations, as_of)
    if as_of_idx is None:
        raise SystemExit(f"No data on or before {as_of.isoformat()}")

    sim_cfg = _build_sim_config(args)
    updates = simulate_distribution_batches(
        features=features,
        as_of_idx=as_of_idx,
        sim_cfg=sim_cfg,
        ma_filter_mode=args.ma_filter_mode,
        batch_size=max(1, args.paths),
        seed=args.seed,
    )
    final = None
    for u in updates:
        final = u
    if final is None:
        raise SystemExit("No simulation output")

    paths = final["paths"]
    checkpoints = sorted({7, 30, 90, 180, 365, sim_cfg.horizon_days})
    checkpoints = [c for c in checkpoints if c <= sim_cfg.horizon_days]
    rows = []
    from sim.mc_core import percentile
    for day in checkpoints:
        prices = [p[day] for p in paths]
        rows.append({"day": day, "price_p05": percentile(prices, 0.05), "price_p50": percentile(prices, 0.50), "price_p95": percentile(prices, 0.95)})
    write_csv(args.summary_out, rows, ["day", "price_p05", "price_p50", "price_p95"])

    anchor = final["anchor"]
    pred = final["prediction"]
    print(f"As-of date: {anchor.obs.date.isoformat()}")
    print(f"Anchor close: {anchor.obs.close:,.2f}")
    print(f"Divergence (%): {anchor.obs.divergence_pct:.2f}")
    print(f"MA({args.ma_window}): {anchor.ma_value if anchor.ma_value is not None else float('nan')}")
    print(f"MA regime: {anchor.ma_regime}")
    print(f"Pool size: {final['bucket_size']}")
    print(f"Fallback used: {final['fallback_used']}")
    print(f"Horizon return p10/p50/p90: {pred['pred_return_p10']:.4f} / {pred['pred_return_p50']:.4f} / {pred['pred_return_p90']:.4f}")
    print(f"Edge score: {pred['pred_return_p50'] - abs(pred['pred_return_p10']):.4f}")
    print(f"Saved summary: {args.summary_out}")


def run_backtest(args: argparse.Namespace) -> None:
    observations = load_observations(args.xlsx)
    features = compute_ma_features(observations, args.ma_window)
    sim_cfg = _build_sim_config(args)

    result = run_backtest_collect(
        features,
        sim_cfg,
        start_date=_parse_date(args.start_date) or features[0].obs.date,
        end_date=_parse_date(args.end_date) or features[-1].obs.date,
        step_days=args.step_days,
        train_lookback_days=args.train_lookback_days,
        ma_filter_mode=args.ma_filter_mode,
        ma_regime_gate=args.ma_regime,
        min_median_return=args.min_median_return,
        min_p10_return=args.min_p10_return,
        max_expected_drawdown=args.max_expected_drawdown,
        non_overlap=args.non_overlap,
        include_baselines=args.include_baselines,
        calibration_bins=args.calibration_bins,
        permutation_runs=args.permutation_runs,
        seed=args.seed,
    )

    rows = result["rows"]
    summary = result["summary"]
    metrics = summary["metrics"]
    write_csv(
        args.out,
        rows,
        [
            "date", "price", "divergence", "ma", "ma_regime", "predicted_return_p10", "predicted_return_p50",
            "predicted_return_p90", "edge_score", "signal", "realized_forward_return", "drawdown_over_horizon",
            "notes", "bucket_size", "equity", "equity_ma_only", "equity_buy_hold"
        ],
    )

    print(f"Backtest rows: {summary['rows']}")
    print(f"Non-overlap mode: {summary['non_overlap']}")
    print(f"Annualized return: {metrics['annualized_return']:.4f}")
    print(f"Annualized volatility: {metrics['annualized_volatility']:.4f}")
    print(f"Sharpe (rf=0): {metrics['sharpe']:.4f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.4f}")
    print(f"Hit rate: {metrics['hit_rate']:.4f}")
    print(f"Average win: {metrics['avg_win']:.4f}")
    print(f"Average loss: {metrics['avg_loss']:.4f}")
    print(f"Fallback share: {summary['fallback_share']:.2%}")
    print(f"Bucket size min/median: {summary['bucket_size_min']}/{summary['bucket_size_median']}")
    print(f"Strict bucket avg realized: {summary['strict_bucket_avg_realized']:.4f}")
    print(f"Fallback bucket avg realized: {summary['fallback_bucket_avg_realized']:.4f}")
    if args.include_baselines:
        b = summary.get("baselines", {})
        print(f"Baseline MA-only Sharpe: {b.get('ma_only', {}).get('sharpe', 0.0):.4f}")
        print(f"Baseline Buy&Hold Sharpe: {b.get('buy_hold', {}).get('sharpe', 0.0):.4f}")
    print(f"Permutation p-value (Sharpe): {summary['permutation_test']['pvalue_sharpe_ge_actual']:.4f}")
    print(f"Saved backtest: {args.out}")

    if args.summary_json:
        write_json(args.summary_json, summary)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--xlsx", default="btc_power_law_2018_plus.xlsx")
    common.add_argument("--paths", type=int, default=5000)
    common.add_argument("--horizon-days", type=int, default=365)
    common.add_argument("--divergence-bandwidth", type=float, default=15.0)
    common.add_argument("--min-bucket-samples", type=int, default=150)
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--ma-window", type=int, default=200)
    common.add_argument("--ma-type", choices=["simple"], default="simple")
    common.add_argument("--ma-regime", choices=["bull", "bear", "none"], default="bull")
    common.add_argument("--ma-filter-mode", choices=["none", "feature", "gate", "both"], default="both")

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    sim = sub.add_parser("simulate", parents=[common])
    sim.add_argument("--as-of-date")
    sim.add_argument("--summary-out", default="mc_summary.csv")

    bt = sub.add_parser("backtest", parents=[common])
    bt.add_argument("--start-date")
    bt.add_argument("--end-date")
    bt.add_argument("--step-days", type=int, default=7)
    bt.add_argument("--train-lookback-days", type=int, default=0)
    bt.add_argument("--decision-rule", choices=["median_p10"], default="median_p10")
    bt.add_argument("--min-median-return", type=float, default=0.0)
    bt.add_argument("--min-p10-return", type=float, default=-0.10)
    bt.add_argument("--max-expected-drawdown", type=float, default=-1.0)
    bt.add_argument("--non-overlap", type=parse_bool, default=True)
    bt.add_argument("--include-baselines", type=parse_bool, default=True)
    bt.add_argument("--calibration-bins", type=int, default=10)
    bt.add_argument("--permutation-runs", type=int, default=200)
    bt.add_argument("--out", default="backtest_results.csv")
    bt.add_argument("--summary-json")
    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) > 1 and sys.argv[1] in {"simulate", "backtest"}:
        args = parser.parse_args()
    else:
        args = parser.parse_args(["simulate", *sys.argv[1:]])

    if args.command == "backtest":
        run_backtest(args)
    else:
        run_simulate(args)


if __name__ == "__main__":
    main()
