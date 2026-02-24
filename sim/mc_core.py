from __future__ import annotations

import csv
import datetime as dt
import json
import math
import random
import statistics
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Sequence, Tuple

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


@dataclass(frozen=True)
class Observation:
    date: dt.date
    close: float
    divergence_pct: float


@dataclass(frozen=True)
class DayFeature:
    obs: Observation
    ma_value: Optional[float]
    ma_regime: str


@dataclass(frozen=True)
class SimulationConfig:
    paths: int
    horizon_days: int
    divergence_bandwidth: float
    min_bucket_samples: int
    seed: Optional[int]


@dataclass(frozen=True)
class BacktestMetrics:
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    avg_win: float
    avg_loss: float


def excel_serial_to_date(serial: float) -> dt.date:
    return dt.date(1899, 12, 30) + dt.timedelta(days=int(serial))


def _column_index(ref: str) -> int:
    letters = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - 64)
    return idx - 1


def load_xlsx_rows(path: Path) -> List[Dict[str, str]]:
    with zipfile.ZipFile(path) as zf:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall(f"{{{MAIN_NS}}}si"):
                shared.append("".join(n.text or "" for n in si.iterfind(f".//{{{MAIN_NS}}}t")))

        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        first_sheet = workbook.find(f"{{{MAIN_NS}}}sheets/{{{MAIN_NS}}}sheet")
        if first_sheet is None:
            raise ValueError("Workbook has no sheets")

        rid = first_sheet.attrib[f"{{{REL_NS}}}id"]
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        target = next((rel.attrib.get("Target") for rel in rels if rel.attrib.get("Id") == rid), None)
        if not target:
            raise ValueError("Could not find first sheet relationship")

        sheet_xml = ET.fromstring(zf.read(f"xl/{target}"))
        rows = sheet_xml.findall(f".//{{{MAIN_NS}}}sheetData/{{{MAIN_NS}}}row")
        if not rows:
            return []

        def parse_row(row: ET.Element) -> Dict[int, str]:
            parsed: Dict[int, str] = {}
            for cell in row.findall(f"{{{MAIN_NS}}}c"):
                cidx = _column_index(cell.attrib.get("r", ""))
                ctype = cell.attrib.get("t")
                value_node = cell.find(f"{{{MAIN_NS}}}v")
                value = value_node.text if value_node is not None and value_node.text is not None else ""
                if ctype == "s" and value:
                    value = shared[int(value)]
                parsed[cidx] = value
            return parsed

        header_map = parse_row(rows[0])
        headers = {idx: name for idx, name in header_map.items() if name}
        out: List[Dict[str, str]] = []
        for row in rows[1:]:
            values = parse_row(row)
            if values:
                out.append({header: values.get(idx, "") for idx, header in headers.items()})
        return out


def to_observations(rows: Sequence[Dict[str, str]]) -> List[Observation]:
    obs: List[Observation] = []
    for row in rows:
        try:
            obs.append(
                Observation(
                    date=excel_serial_to_date(float(row["Date"])),
                    close=float(row["Close"]),
                    divergence_pct=float(row["PL_Divergence_Pct"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    obs.sort(key=lambda x: x.date)
    return obs


def load_observations(xlsx_path: str | Path) -> List[Observation]:
    return to_observations(load_xlsx_rows(Path(xlsx_path)))


def resolve_as_of_idx(observations: Sequence[Observation], as_of_date: dt.date) -> Optional[int]:
    for i in range(len(observations) - 1, -1, -1):
        if observations[i].date <= as_of_date:
            return i
    return None


def compute_ma_features(observations: Sequence[Observation], ma_window: int) -> List[DayFeature]:
    if ma_window <= 0:
        raise ValueError("ma_window must be > 0")
    rolling: List[float] = []
    rolling_sum = 0.0
    out: List[DayFeature] = []
    for obs in observations:
        rolling.append(obs.close)
        rolling_sum += obs.close
        if len(rolling) > ma_window:
            rolling_sum -= rolling.pop(0)
        ma = (rolling_sum / ma_window) if len(rolling) >= ma_window else None
        regime = "unknown" if ma is None else ("bull" if obs.close >= ma else "bear")
        out.append(DayFeature(obs=obs, ma_value=ma, ma_regime=regime))
    return out


def build_return_samples(features: Sequence[DayFeature]) -> List[Tuple[DayFeature, float]]:
    samples: List[Tuple[DayFeature, float]] = []
    for i in range(len(features) - 1):
        now, nxt = features[i], features[i + 1]
        if now.obs.close > 0:
            samples.append((now, (nxt.obs.close / now.obs.close) - 1.0))
    return samples


def _mean_reversion_weight(sample_div: float, current_div: float) -> float:
    return 1.0 / (1.0 + abs(sample_div - current_div) / 25.0)


def _ma_pass(sample_regime: str, current_regime: str, ma_filter_mode: str) -> bool:
    if ma_filter_mode in {"none", "feature"}:
        return True
    if current_regime in {"none", "unknown"}:
        return False
    return sample_regime == current_regime


def choose_pool(
    samples: Sequence[Tuple[DayFeature, float]],
    current_divergence: float,
    current_regime: str,
    bandwidth: float,
    min_bucket_samples: int,
    ma_filter_mode: str,
) -> Tuple[List[Tuple[float, float]], str]:
    strict = [
        (ret, _mean_reversion_weight(feat.obs.divergence_pct, current_divergence))
        for feat, ret in samples
        if abs(feat.obs.divergence_pct - current_divergence) <= bandwidth
        and _ma_pass(feat.ma_regime, current_regime, ma_filter_mode)
    ]
    if len(strict) >= min_bucket_samples:
        return strict, "strict_divergence+ma"

    ma_only = [
        (ret, _mean_reversion_weight(feat.obs.divergence_pct, current_divergence))
        for feat, ret in samples
        if _ma_pass(feat.ma_regime, current_regime, ma_filter_mode)
    ]
    if len(ma_only) >= min_bucket_samples:
        return ma_only, "ma_only"

    return [
        (ret, _mean_reversion_weight(feat.obs.divergence_pct, current_divergence))
        for feat, ret in samples
    ], "all_history"


def weighted_pick(pool: Sequence[Tuple[float, float]], rng: random.Random) -> float:
    total = sum(max(w, 1e-9) for _, w in pool)
    target = rng.random() * total
    run = 0.0
    for value, w in pool:
        run += max(w, 1e-9)
        if run >= target:
            return value
    return pool[-1][0]


def simulate_paths(current_price: float, pool: Sequence[Tuple[float, float]], config: SimulationConfig, rng: random.Random) -> List[List[float]]:
    all_paths: List[List[float]] = []
    for _ in range(config.paths):
        price = current_price
        path = [price]
        for _ in range(config.horizon_days):
            price *= 1.0 + weighted_pick(pool, rng)
            path.append(price)
        all_paths.append(path)
    return all_paths


def percentile(values: Sequence[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = (len(ordered) - 1) * pct
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize_returns(paths: Sequence[Sequence[float]], horizon_days: int) -> Dict[str, float]:
    rets = [(p[horizon_days] / p[0]) - 1.0 for p in paths]
    p10 = percentile(rets, 0.10)
    p50 = percentile(rets, 0.50)
    return {
        "pred_return_p10": p10,
        "pred_return_p50": p50,
        "pred_return_p90": percentile(rets, 0.90),
        "pred_return_mean": statistics.fmean(rets),
        "edge_score": p50 - abs(p10),
    }


def calc_drawdown(path: Sequence[float]) -> float:
    peak = path[0]
    max_dd = 0.0
    for value in path:
        peak = max(peak, value)
        max_dd = min(max_dd, (value / peak) - 1.0)
    return max_dd


def evaluate_at_index(
    features: Sequence[DayFeature],
    index: int,
    sim_cfg: SimulationConfig,
    rng: random.Random,
    ma_filter_mode: str,
    train_lookback_days: int,
    require_forward: bool = True,
) -> Optional[Dict[str, object]]:
    has_forward_window = index + sim_cfg.horizon_days < len(features)
    if require_forward and not has_forward_window:
        return None
    start = 0
    if train_lookback_days > 0:
        min_date = features[index].obs.date - dt.timedelta(days=train_lookback_days)
        start = index
        while start > 0 and features[start - 1].obs.date >= min_date:
            start -= 1

    history = features[start : index + 1]
    samples = build_return_samples(history)
    if not samples:
        return None

    anchor = features[index]
    pool, fallback_used = choose_pool(
        samples=samples,
        current_divergence=anchor.obs.divergence_pct,
        current_regime=anchor.ma_regime,
        bandwidth=sim_cfg.divergence_bandwidth,
        min_bucket_samples=sim_cfg.min_bucket_samples,
        ma_filter_mode=ma_filter_mode,
    )

    paths = simulate_paths(anchor.obs.close, pool, sim_cfg, rng)
    prediction = summarize_returns(paths, sim_cfg.horizon_days)
    expected_dd = statistics.fmean(calc_drawdown(p) for p in paths)
    realized = None
    if has_forward_window:
        realized = (features[index + sim_cfg.horizon_days].obs.close / anchor.obs.close) - 1.0
    return {
        "anchor": anchor,
        "pool": pool,
        "fallback_used": fallback_used,
        "prediction": prediction,
        "expected_drawdown": expected_dd,
        "realized_forward_return": realized,
        "paths": paths,
        "history_start": history[0].obs.date,
        "history_end": history[-1].obs.date,
        "training_pool_size": len(samples),
    }


def simulate_distribution_batches(
    features: Sequence[DayFeature],
    as_of_idx: int,
    sim_cfg: SimulationConfig,
    ma_filter_mode: str,
    batch_size: int,
    seed: Optional[int],
) -> Generator[Dict[str, object], None, Dict[str, object]]:
    rng = random.Random(seed)
    eval_cfg = SimulationConfig(
        paths=1,
        horizon_days=sim_cfg.horizon_days,
        divergence_bandwidth=sim_cfg.divergence_bandwidth,
        min_bucket_samples=sim_cfg.min_bucket_samples,
        seed=seed,
    )
    state = evaluate_at_index(features, as_of_idx, eval_cfg, rng, ma_filter_mode, 0, require_forward=False)
    if state is None:
        raise RuntimeError("Failed to evaluate state")

    pool = state["pool"]
    anchor: DayFeature = state["anchor"]  # type: ignore[assignment]
    all_paths: List[List[float]] = []
    generated = 0
    while generated < sim_cfg.paths:
        this_batch = min(batch_size, sim_cfg.paths - generated)
        batch_cfg = SimulationConfig(
            paths=this_batch,
            horizon_days=sim_cfg.horizon_days,
            divergence_bandwidth=sim_cfg.divergence_bandwidth,
            min_bucket_samples=sim_cfg.min_bucket_samples,
            seed=seed,
        )
        new_paths = simulate_paths(anchor.obs.close, pool, batch_cfg, rng)
        all_paths.extend(new_paths)
        generated += this_batch
        rets = [(p[sim_cfg.horizon_days] / p[0]) - 1.0 for p in all_paths]
        yield {
            "generated_paths": generated,
            "total_paths": sim_cfg.paths,
            "prediction": {
                "pred_return_p10": percentile(rets, 0.10),
                "pred_return_p50": percentile(rets, 0.50),
                "pred_return_p90": percentile(rets, 0.90),
            },
            "paths": list(all_paths),
            "anchor": anchor,
            "fallback_used": state["fallback_used"],
            "bucket_size": len(pool),
            "divergence": anchor.obs.divergence_pct,
            "ma_value": anchor.ma_value,
            "ma_regime": anchor.ma_regime,
            "training_pool_size": state["training_pool_size"],
            "history_start": state["history_start"],
            "history_end": state["history_end"],
        }
    return state


def decision_signal(
    ma_regime: str,
    gate_regime: str,
    pred: Dict[str, float],
    min_median_return: float,
    min_p10_return: float,
    max_expected_drawdown: float,
    expected_drawdown: float,
) -> str:
    if gate_regime != "none" and ma_regime != gate_regime:
        return "flat"
    if expected_drawdown < max_expected_drawdown:
        return "flat"
    if pred["pred_return_p50"] >= min_median_return and pred["pred_return_p10"] >= min_p10_return:
        return "long"
    return "flat"


def backtest_metrics(equity_curve: Sequence[float], step_returns: Sequence[float], step_days: int) -> BacktestMetrics:
    if len(equity_curve) < 2:
        return BacktestMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    periods_per_year = 365.0 / max(step_days, 1)
    avg = statistics.fmean(step_returns) if step_returns else 0.0
    vol = statistics.pstdev(step_returns) if len(step_returns) > 1 else 0.0
    ann_ret = (1.0 + avg) ** periods_per_year - 1.0
    ann_vol = vol * math.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        max_dd = min(max_dd, (eq / peak) - 1.0)

    wins = [r for r in step_returns if r > 0]
    losses = [r for r in step_returns if r < 0]
    return BacktestMetrics(
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe=sharpe,
        max_drawdown=max_dd,
        hit_rate=(len(wins) / len(step_returns)) if step_returns else 0.0,
        avg_win=statistics.fmean(wins) if wins else 0.0,
        avg_loss=statistics.fmean(losses) if losses else 0.0,
    )


def equity_from_returns(returns: Sequence[float]) -> List[float]:
    eq = [1.0]
    for r in returns:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def bin_calibration(rows: Sequence[Dict[str, object]], bins: int = 10) -> List[Dict[str, float]]:
    if not rows:
        return []
    ordered = sorted(rows, key=lambda r: float(r["edge_score"]))
    n = len(ordered)
    out: List[Dict[str, float]] = []
    for b in range(bins):
        lo = int((b / bins) * n)
        hi = int(((b + 1) / bins) * n)
        if lo >= hi:
            continue
        chunk = ordered[lo:hi]
        out.append(
            {
                "bin": b + 1,
                "count": len(chunk),
                "avg_edge": statistics.fmean(float(r["edge_score"]) for r in chunk),
                "avg_realized": statistics.fmean(float(r["realized_forward_return"]) for r in chunk),
            }
        )
    return out


def permutation_test(
    long_flags: Sequence[bool],
    realized_returns: Sequence[float],
    actual_sharpe: float,
    runs: int,
    seed: int,
    step_days: int,
) -> Dict[str, float]:
    if runs <= 0 or len(realized_returns) < 2 or len(long_flags) != len(realized_returns):
        return {"runs": 0, "pvalue_sharpe_ge_actual": 1.0, "null_sharpe_p95": 0.0}

    rng = random.Random(seed + 999)
    null_sharpes: List[float] = []
    for _ in range(runs):
        shuffled = list(realized_returns)
        rng.shuffle(shuffled)
        strat = [ret if is_long else 0.0 for is_long, ret in zip(long_flags, shuffled)]
        m = backtest_metrics(equity_from_returns(strat), strat, step_days)
        null_sharpes.append(m.sharpe)

    pval = sum(1 for s in null_sharpes if s >= actual_sharpe) / len(null_sharpes)
    return {
        "runs": runs,
        "pvalue_sharpe_ge_actual": pval,
        "null_sharpe_p95": percentile(null_sharpes, 0.95),
    }


def run_backtest_collect(
    features: Sequence[DayFeature],
    sim_cfg: SimulationConfig,
    *,
    start_date: dt.date,
    end_date: dt.date,
    step_days: int,
    train_lookback_days: int,
    ma_filter_mode: str,
    ma_regime_gate: str,
    min_median_return: float,
    min_p10_return: float,
    max_expected_drawdown: float,
    non_overlap: bool,
    include_baselines: bool,
    calibration_bins: int,
    permutation_runs: int,
    seed: int,
) -> Dict[str, object]:
    rng = random.Random(seed)
    idxs = [i for i, f in enumerate(features) if start_date <= f.obs.date <= end_date and i + sim_cfg.horizon_days < len(features)]
    idxs = [i for i in idxs if ((features[i].obs.date - start_date).days % step_days == 0)]
    if non_overlap:
        filtered: List[int] = []
        next_allowed = 0
        for i in idxs:
            if i >= next_allowed:
                filtered.append(i)
                next_allowed = i + sim_cfg.horizon_days
        idxs = filtered

    rows: List[Dict[str, object]] = []
    strat_returns: List[float] = []
    ma_returns: List[float] = []
    bh_returns: List[float] = []

    for i in idxs:
        state = evaluate_at_index(features, i, sim_cfg, rng, ma_filter_mode, train_lookback_days)
        if state is None:
            continue
        anchor: DayFeature = state["anchor"]  # type: ignore[assignment]
        pred: Dict[str, float] = state["prediction"]  # type: ignore[assignment]
        realized = float(state["realized_forward_return"])
        expected_dd = float(state["expected_drawdown"])
        signal = decision_signal(anchor.ma_regime, ma_regime_gate, pred, min_median_return, min_p10_return, max_expected_drawdown, expected_dd)

        strat_ret = realized if signal == "long" else 0.0
        ma_ret = realized if (ma_regime_gate == "none" or anchor.ma_regime == ma_regime_gate) else 0.0
        bh_ret = realized
        strat_returns.append(strat_ret)
        ma_returns.append(ma_ret)
        bh_returns.append(bh_ret)

        rows.append(
            {
                "date": anchor.obs.date.isoformat(),
                "price": anchor.obs.close,
                "divergence": anchor.obs.divergence_pct,
                "ma": "" if anchor.ma_value is None else anchor.ma_value,
                "ma_regime": anchor.ma_regime,
                "predicted_return_p10": pred["pred_return_p10"],
                "predicted_return_p50": pred["pred_return_p50"],
                "predicted_return_p90": pred["pred_return_p90"],
                "edge_score": pred["edge_score"],
                "signal": signal,
                "realized_forward_return": realized,
                "drawdown_over_horizon": expected_dd,
                "notes": state["fallback_used"],
                "bucket_size": len(state["pool"]),
            }
        )

    strat_eq = equity_from_returns(strat_returns)
    ma_eq = equity_from_returns(ma_returns)
    bh_eq = equity_from_returns(bh_returns)
    for i, row in enumerate(rows):
        row["equity"] = strat_eq[i + 1]
        row["equity_ma_only"] = ma_eq[i + 1]
        row["equity_buy_hold"] = bh_eq[i + 1]

    effective_step = sim_cfg.horizon_days if non_overlap and step_days < sim_cfg.horizon_days else step_days
    metrics = backtest_metrics(strat_eq, strat_returns, effective_step)
    baselines = {
        "ma_only": backtest_metrics(ma_eq, ma_returns, effective_step),
        "buy_hold": backtest_metrics(bh_eq, bh_returns, effective_step),
    }

    fallback_rows = [r for r in rows if str(r["notes"]) != "strict_divergence+ma"]
    strict_rows = [r for r in rows if str(r["notes"]) == "strict_divergence+ma"]
    bucket_sizes = [int(r["bucket_size"]) for r in rows]
    perm = permutation_test([str(r["signal"]) == "long" for r in rows], [float(r["realized_forward_return"]) for r in rows], metrics.sharpe, permutation_runs, seed, effective_step)

    summary: Dict[str, object] = {
        "rows": len(rows),
        "non_overlap": non_overlap,
        "metrics": metrics.__dict__,
        "fallback_share": (len(fallback_rows) / len(rows)) if rows else 0.0,
        "bucket_size_min": min(bucket_sizes) if bucket_sizes else 0,
        "bucket_size_median": statistics.median(bucket_sizes) if bucket_sizes else 0,
        "strict_bucket_avg_realized": statistics.fmean(float(r["realized_forward_return"]) for r in strict_rows) if strict_rows else 0.0,
        "fallback_bucket_avg_realized": statistics.fmean(float(r["realized_forward_return"]) for r in fallback_rows) if fallback_rows else 0.0,
        "calibration_bins": bin_calibration(rows, bins=calibration_bins),
        "permutation_test": perm,
    }
    if include_baselines:
        summary["baselines"] = {k: v.__dict__ for k, v in baselines.items()}

    return {
        "rows": rows,
        "summary": summary,
        "metrics": metrics,
        "baselines": baselines,
        "fallback_rows": fallback_rows,
        "strict_rows": strict_rows,
    }


def run_backtest_batches(
    features: Sequence[DayFeature],
    sim_cfg: SimulationConfig,
    *,
    start_date: dt.date,
    end_date: dt.date,
    step_days: int,
    train_lookback_days: int,
    ma_filter_mode: str,
    ma_regime_gate: str,
    min_median_return: float,
    min_p10_return: float,
    max_expected_drawdown: float,
    non_overlap: bool,
    include_baselines: bool,
    calibration_bins: int,
    permutation_runs: int,
    seed: int,
    batch_eval: int = 5,
) -> Generator[Dict[str, object], None, Dict[str, object]]:
    rng = random.Random(seed)
    idxs = [i for i, f in enumerate(features) if start_date <= f.obs.date <= end_date and i + sim_cfg.horizon_days < len(features)]
    idxs = [i for i in idxs if ((features[i].obs.date - start_date).days % step_days == 0)]
    if non_overlap:
        filtered: List[int] = []
        next_allowed = 0
        for i in idxs:
            if i >= next_allowed:
                filtered.append(i)
                next_allowed = i + sim_cfg.horizon_days
        idxs = filtered

    rows: List[Dict[str, object]] = []
    strat_returns: List[float] = []
    ma_returns: List[float] = []
    bh_returns: List[float] = []

    total = len(idxs)
    for pos, i in enumerate(idxs, start=1):
        state = evaluate_at_index(features, i, sim_cfg, rng, ma_filter_mode, train_lookback_days)
        if state is None:
            continue
        anchor: DayFeature = state["anchor"]  # type: ignore[assignment]
        pred: Dict[str, float] = state["prediction"]  # type: ignore[assignment]
        realized = float(state["realized_forward_return"])
        expected_dd = float(state["expected_drawdown"])
        signal = decision_signal(anchor.ma_regime, ma_regime_gate, pred, min_median_return, min_p10_return, max_expected_drawdown, expected_dd)

        strat_ret = realized if signal == "long" else 0.0
        ma_ret = realized if (ma_regime_gate == "none" or anchor.ma_regime == ma_regime_gate) else 0.0
        bh_ret = realized
        strat_returns.append(strat_ret)
        ma_returns.append(ma_ret)
        bh_returns.append(bh_ret)

        row = {
            "date": anchor.obs.date.isoformat(),
            "price": anchor.obs.close,
            "divergence": anchor.obs.divergence_pct,
            "ma": "" if anchor.ma_value is None else anchor.ma_value,
            "ma_regime": anchor.ma_regime,
            "predicted_return_p10": pred["pred_return_p10"],
            "predicted_return_p50": pred["pred_return_p50"],
            "predicted_return_p90": pred["pred_return_p90"],
            "edge_score": pred["edge_score"],
            "signal": signal,
            "realized_forward_return": realized,
            "drawdown_over_horizon": expected_dd,
            "notes": state["fallback_used"],
            "bucket_size": len(state["pool"]),
        }
        rows.append(row)

        if pos % max(batch_eval, 1) == 0 or pos == total:
            eq = equity_from_returns(strat_returns)
            for j, r in enumerate(rows):
                r["equity"] = eq[j + 1]
            yield {"progress": pos, "total": total, "rows": list(rows)}

    strat_eq = equity_from_returns(strat_returns)
    ma_eq = equity_from_returns(ma_returns)
    bh_eq = equity_from_returns(bh_returns)
    for i, row in enumerate(rows):
        row["equity"] = strat_eq[i + 1]
        row["equity_ma_only"] = ma_eq[i + 1]
        row["equity_buy_hold"] = bh_eq[i + 1]

    effective_step = sim_cfg.horizon_days if non_overlap and step_days < sim_cfg.horizon_days else step_days
    metrics = backtest_metrics(strat_eq, strat_returns, effective_step)
    baselines = {
        "ma_only": backtest_metrics(ma_eq, ma_returns, effective_step),
        "buy_hold": backtest_metrics(bh_eq, bh_returns, effective_step),
    }

    fallback_rows = [r for r in rows if str(r["notes"]) != "strict_divergence+ma"]
    strict_rows = [r for r in rows if str(r["notes"]) == "strict_divergence+ma"]
    bucket_sizes = [int(r["bucket_size"]) for r in rows]
    perm = permutation_test([str(r["signal"]) == "long" for r in rows], [float(r["realized_forward_return"]) for r in rows], metrics.sharpe, permutation_runs, seed, effective_step)

    summary: Dict[str, object] = {
        "rows": len(rows),
        "non_overlap": non_overlap,
        "metrics": metrics.__dict__,
        "fallback_share": (len(fallback_rows) / len(rows)) if rows else 0.0,
        "bucket_size_min": min(bucket_sizes) if bucket_sizes else 0,
        "bucket_size_median": statistics.median(bucket_sizes) if bucket_sizes else 0,
        "strict_bucket_avg_realized": statistics.fmean(float(r["realized_forward_return"]) for r in strict_rows) if strict_rows else 0.0,
        "fallback_bucket_avg_realized": statistics.fmean(float(r["realized_forward_return"]) for r in fallback_rows) if fallback_rows else 0.0,
        "calibration_bins": bin_calibration(rows, bins=calibration_bins),
        "permutation_test": perm,
    }
    if include_baselines:
        summary["baselines"] = {k: v.__dict__ for k, v in baselines.items()}

    return {
        "rows": rows,
        "summary": summary,
        "metrics": metrics,
        "baselines": baselines,
        "fallback_rows": fallback_rows,
        "strict_rows": strict_rows,
    }


def write_csv(path: str | Path, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    p = Path(path)
    if p.parent != Path(""):
        p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Dict[str, object]) -> None:
    p = Path(path)
    if p.parent != Path(""):
        p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
