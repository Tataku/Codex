from __future__ import annotations

import csv
import datetime as dt
import io
import json
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from sim.mc_core import (
    SimulationConfig,
    compute_ma_features,
    load_observations,
    percentile,
    run_backtest_batches,
    simulate_distribution_batches,
)

st.set_page_config(page_title="Simulator UI", layout="wide")
st.title("BTC Monte Carlo Simulator UI")

with st.sidebar:
    mode = st.selectbox("Mode", ["Simulate", "Backtest"])
    xlsx = st.text_input("XLSX path", "btc_power_law_2018_plus.xlsx")
    paths = int(st.number_input("paths", min_value=1, value=5000))
    horizon_days = int(st.number_input("horizon_days", min_value=1, value=365))
    seed = int(st.number_input("seed", value=42))
    divergence_bandwidth = float(st.number_input("divergence_bandwidth", value=15.0))
    min_bucket_samples = int(st.number_input("min_bucket_samples", min_value=1, value=150))
    ma_window = int(st.number_input("ma-window", min_value=1, value=200))
    ma_type = st.selectbox("ma-type", ["simple"], index=0)
    ma_regime = st.selectbox("ma-regime", ["bull", "bear", "none"], index=0)
    ma_filter_mode = st.selectbox("ma-filter-mode", ["none", "feature", "gate", "both"], index=3)

if not Path(xlsx).exists():
    st.error(f"XLSX path not found: {xlsx}")
    st.stop()

observations = load_observations(xlsx)
features = compute_ma_features(observations, ma_window)
sim_cfg = SimulationConfig(paths=paths, horizon_days=horizon_days, divergence_bandwidth=divergence_bandwidth, min_bucket_samples=min_bucket_samples, seed=seed)

if mode == "Simulate":
    as_of = st.date_input("as_of_date", value=features[-1].obs.date)
    show_spaghetti = st.checkbox("show_spaghetti", value=True)
    max_plotted_paths = int(st.number_input("max_plotted_paths", min_value=10, value=150))
    batch_size = int(st.number_input("batch_size", min_value=1, value=50))

    if st.button("Run simulate"):
        as_of_idx = next((i for i in range(len(features) - 1, -1, -1) if features[i].obs.date <= as_of), None)
        if as_of_idx is None:
            st.error("No data available for selected date")
            st.stop()

        progress = st.progress(0)
        diag = st.empty()
        fan_slot = st.empty()
        spaghetti_slot = st.empty()
        pred_slot = st.empty()

        final_update = None
        for update in simulate_distribution_batches(features, as_of_idx, sim_cfg, ma_filter_mode, batch_size, seed):
            final_update = update
            done = int(update["generated_paths"])
            total = int(update["total_paths"])
            progress.progress(min(done / total, 1.0))

            paths_so_far = update["paths"]
            p10, p50, p90 = [], [], []
            for day in range(horizon_days + 1):
                vals = [p[day] for p in paths_so_far]
                p10.append(percentile(vals, 0.10))
                p50.append(percentile(vals, 0.50))
                p90.append(percentile(vals, 0.90))
            fan_slot.line_chart({"p10": p10, "p50": p50, "p90": p90})

            if show_spaghetti and paths_so_far:
                subset = paths_so_far[:max_plotted_paths]
                spaghetti_slot.line_chart({f"path_{i}": p for i, p in enumerate(subset[:10])})

            diag.info(
                f"progress={done}/{total} | bucket={update['bucket_size']} | fallback={update['fallback_used']} | "
                f"div={update['divergence']:.2f}% | ma={update['ma_value']} | regime={update['ma_regime']} | "
                f"history={update['history_start']}..{update['history_end']} | train_pool={update['training_pool_size']}"
            )
            pred_slot.dataframe([update["prediction"]], use_container_width=True)

        if final_update:
            paths_final = final_update["paths"]
            summary_rows = []
            for day in range(horizon_days + 1):
                vals = [p[day] for p in paths_final]
                summary_rows.append({"day": day, "p10": percentile(vals, 0.10), "p50": percentile(vals, 0.50), "p90": percentile(vals, 0.90)})

            b1 = io.StringIO()
            w1 = csv.DictWriter(b1, fieldnames=["day", "p10", "p50", "p90"])
            w1.writeheader()
            w1.writerows(summary_rows)
            st.download_button("Download summary CSV", b1.getvalue(), "mc_summary_ui.csv", "text/csv")

            b2 = io.StringIO()
            w2 = csv.writer(b2)
            w2.writerow(["path_id", "day", "price"])
            for pid, path in enumerate(paths_final):
                for day, price in enumerate(path):
                    w2.writerow([pid, day, price])
            st.download_button("Download paths CSV", b2.getvalue(), "mc_paths_ui.csv", "text/csv")

else:
    start_date = st.date_input("start_date", value=features[0].obs.date)
    end_date = st.date_input("end_date", value=features[-1].obs.date)
    step_days = int(st.number_input("step_days", min_value=1, value=7))
    train_lookback_days = int(st.number_input("train_lookback_days", min_value=0, value=0))
    min_median_return = float(st.number_input("min-median-return", value=0.0))
    min_p10_return = float(st.number_input("min-p10-return", value=-0.10))
    max_expected_drawdown = float(st.number_input("max-expected-drawdown", value=-1.0))
    non_overlap = st.checkbox("non-overlap", value=True)
    include_baselines = st.checkbox("include-baselines", value=True)
    calibration_bins = int(st.number_input("calibration-bins", min_value=1, value=10))
    permutation_runs = int(st.number_input("permutation-runs", min_value=0, value=200))
    batch_eval = int(st.number_input("batch_eval", min_value=1, value=5))

    if st.button("Run backtest"):
        progress = st.progress(0)
        eq_slot = st.empty()
        dd_slot = st.empty()
        metrics_slot = st.empty()
        table_slot = st.empty()

        gen = run_backtest_batches(
            features,
            sim_cfg,
            start_date=start_date,
            end_date=end_date,
            step_days=step_days,
            train_lookback_days=train_lookback_days,
            ma_filter_mode=ma_filter_mode,
            ma_regime_gate=ma_regime,
            min_median_return=min_median_return,
            min_p10_return=min_p10_return,
            max_expected_drawdown=max_expected_drawdown,
            non_overlap=non_overlap,
            include_baselines=include_baselines,
            calibration_bins=calibration_bins,
            permutation_runs=permutation_runs,
            seed=seed,
            batch_eval=batch_eval,
        )

        final = None
        while True:
            try:
                update = next(gen)
            except StopIteration as stop:
                final = stop.value
                break
            total = int(update["total"])
            prog = int(update["progress"])
            rows = update["rows"]
            if total:
                progress.progress(min(prog / total, 1.0))
            equities = [float(r.get("equity", 1.0)) for r in rows]
            eq_slot.line_chart(equities)
            peak = 1.0
            dds = []
            for e in equities:
                peak = max(peak, e)
                dds.append((e / peak) - 1.0)
            dd_slot.area_chart(dds)
            table_slot.dataframe(rows[-20:], use_container_width=True)

        if not final:
            st.error("No backtest output")
            st.stop()

        rows = final["rows"]
        summary = final["summary"]
        metrics_slot.json(summary)
        table_slot.dataframe(rows, use_container_width=True)

        b1 = io.StringIO()
        if rows:
            w = csv.DictWriter(b1, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        st.download_button("Download backtest CSV", b1.getvalue(), "backtest_ui.csv", "text/csv")
        st.download_button("Download backtest summary JSON", json.dumps(summary, indent=2), "backtest_ui_summary.json", "application/json")
