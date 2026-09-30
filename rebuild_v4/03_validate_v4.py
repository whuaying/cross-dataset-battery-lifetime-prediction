"""Independent consistency gates for the v4 residual-life analysis."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom, lognorm

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "rebuild_v2"
MODELS = {"km", "xgb_aft", "lognormal_aft", "cox", "rsf"}


def csv_rows(name: str) -> list[dict]:
    with (HERE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1, abs(a), abs(b))


def main() -> None:
    protocol = json.loads((HERE / "protocol_v4.json").read_text(encoding="utf-8"))
    for name, expected in protocol["input_sha256"].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise AssertionError(f"Frozen input changed: {name}")
    if protocol["event_time_model"] != "residual time R=T-L in the cohort event-free at landmark L":
        raise AssertionError("Time estimand changed")
    budget = csv_rows("calibration_budget_feasibility.csv")
    if len(budget) != 2 * 6 * 3:
        raise AssertionError("Budget grid incomplete")
    for r in budget:
        n, k, m = int(r["n_target"]), int(r["n_events"]), int(r["n_calibration"])
        if m != math.ceil(float(r["requested_fraction"]) * n):
            raise AssertionError("Budget round-up mismatch")
        probability = float(r["probability_of_enough_events_random_draw"])
        if not close(probability, float(hypergeom.sf(18, n, k, m))):
            raise AssertionError("Conformal label-count probability mismatch")
        if int(r["finite_quantile_possible_for_some_draw"]) != int(m >= 19 and k >= 19):
            raise AssertionError("Conformal existence flag mismatch")

    benchmark = csv_rows("benchmark_predictions.csv")
    source_selection = csv_rows("source_validation_results.csv")
    expected_cells = {}
    for landmark in (50, 100):
        with (SOURCE / f"landmark_{landmark}_features.csv").open(encoding="utf-8-sig", newline="") as stream:
            for r in csv.DictReader(stream):
                expected_cells[(landmark, r["dataset"], r["cell_id"])] = r
    selected = defaultdict(set)
    for r in source_selection:
        if r["inner_validation_domain"] == r["held_out_domain"]:
            raise AssertionError("Target labels used in source model selection")
        selected[(int(r["landmark"]), r["held_out_domain"], r["model"],
                  r["candidate"])].add(r["inner_validation_domain"])
    if len(selected) != 2 * 6 * (4 + 3 + 3 + 2) or any(len(v) != 5 for v in selected.values()):
        raise AssertionError("Source-only inner validation incomplete")
    prior = json.loads((HERE / "source_prior_selection.json").read_text(encoding="utf-8"))
    prior_rows = csv_rows("source_prior_validation.csv")
    if len(prior_rows) != 2 * 6 * 5 * 10 * 3 or prior["outer_target_outcomes_used"]:
        raise AssertionError("Source-only target-shift selection incomplete")
    scores_by_prior = defaultdict(list)
    for r in prior_rows:
        if r["source_validation_domain"] == r["outer_held_out_domain"]:
            raise AssertionError("Outer target used to select shift prior")
        scores_by_prior[(r["landmark"], r["outer_held_out_domain"], r["prior_log_shift_sd"])].append(
            float(r["evaluation_mean_censored_nll"]))
    for landmark in ("50", "100"):
        for domain in protocol["domains"]:
            candidates = {p: np.mean(scores_by_prior[(landmark, domain, str(p))])
                          for p in (0.25, 0.5, 1.0)}
            if not close(float(prior["selected_prior_log_sd"][landmark][domain]),
                         min(candidates, key=candidates.get)):
                raise AssertionError("Selected shift prior disagrees with source NLL")
    model_cells = defaultdict(set)
    for r in benchmark:
        landmark = int(r["landmark"])
        key = (landmark, r["held_out_domain"], r["cell_id"])
        source = expected_cells[key]
        if not close(float(r["residual_followup"]) + landmark, float(source["followup_cycle"])):
            raise AssertionError("Residual time not equal to total minus landmark")
        if int(r["event_observed"]) != int(source["event_observed"]):
            raise AssertionError("Event flag changed")
        role = (landmark, r["held_out_domain"], r["model"])
        if r["cell_id"] in model_cells[role]:
            raise AssertionError("Duplicate prediction")
        model_cells[role].add(r["cell_id"])
        risks = np.array([float(r[f"risk_by_{t}"]) for t in protocol["horizons_absolute_cycle"] if t > landmark])
        if not np.all((risks >= -1e-10) & (risks <= 1 + 1e-10)):
            raise AssertionError("Probability outside [0,1]")
        if np.any(np.diff(risks) < -1e-9):
            raise AssertionError("Event probability decreases over time")
        if r["model"] in {"xgb_aft", "lognormal_aft"}:
            mu, sigma = float(r["aft_mu"]), float(r["aft_sigma"])
            median = float(r["median_total_cycle"])
            if not close(median, landmark + math.exp(mu)):
                raise AssertionError("AFT median does not use residual time")
            for horizon in protocol["horizons_absolute_cycle"]:
                if horizon > landmark:
                    independently_computed = float(lognorm.cdf(horizon - landmark, s=sigma, scale=math.exp(mu)))
                    if not close(float(r[f"risk_by_{horizon}"]), independently_computed):
                        raise AssertionError("AFT risk differs from independent lognormal CDF")
    if len(model_cells) != 2 * 6 * 5 or sum(len(cells) for cells in model_cells.values()) != 5 * len(expected_cells):
        raise AssertionError("Five models do not cover every held-out cell")
    if {r["model"] for r in benchmark} != MODELS:
        raise AssertionError("Benchmark model set changed")

    scores = csv_rows("recalibration_scores.csv")
    membership = csv_rows("recalibration_membership.csv")
    if len(scores) != 2 * 6 * 3 * 30:
        raise AssertionError("Target recalibration splits incomplete")
    grouped = defaultdict(list)
    for r in membership:
        grouped[(int(r["landmark"]), r["domain"], r["budget_fraction"], int(r["split"]))].append(r)
    if len(grouped) != len(scores):
        raise AssertionError("Membership missing a split")
    benchmark_lookup = {(int(r["landmark"]), r["held_out_domain"], r["cell_id"]): r
                        for r in benchmark if r["model"] == "xgb_aft"}
    for r in scores:
        key = (int(r["landmark"]), r["domain"], r["budget_fraction"], int(r["split"]))
        members = grouped[key]
        ids = [m["cell_id"] for m in members]
        if len(ids) != len(set(ids)) or set(ids) != model_cells[(key[0], key[1], "xgb_aft")]:
            raise AssertionError("Target partition drops or repeats a cell")
        cal = [m for m in members if m["role"] == "calibration"]
        test = [m for m in members if m["role"] == "evaluation"]
        if len(cal) != math.ceil(float(key[2]) * len(members)) or len(test) != len(members) - len(cal):
            raise AssertionError("Target budget does not use ceil(p*n)")
        if len(cal) != int(r["n_calibration"]) or len(test) != int(r["n_evaluation"]):
            raise AssertionError("Target partition size disagrees with score")
        if not close(float(r["prior_log_shift_sd"]),
                     float(prior["selected_prior_log_sd"][str(key[0])][key[1]])):
            raise AssertionError("Target shift did not use source-selected prior")
        if sum(int(m["event_observed"]) for m in cal) != int(r["n_calibration_events"]):
            raise AssertionError("Target event count disagrees")
        if not math.isfinite(float(r["nll_difference"])):
            raise AssertionError("Non-finite recalibration score")
        for member in members:
            p = benchmark_lookup[(key[0], key[1], member["cell_id"])]
            if not close(float(member["baseline_mu"]), float(p["aft_mu"])):
                raise AssertionError("Target prediction not frozen")
            if not close(float(member["residual_followup"]), float(p["residual_followup"])):
                raise AssertionError("Target follow-up disagrees")
    print("PASS: v4 input hashes, landmark time axis, source-only model and shift-prior selection, five-model partition, independent AFT CDF, monotone risks, target split disjointness, and exact budget probabilities")


if __name__ == "__main__":
    main()
