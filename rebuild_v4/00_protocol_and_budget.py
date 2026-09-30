"""Freeze the v4 analysis protocol and correct target-label budget feasibility."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import hypergeom

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "rebuild_v2"
LANDMARKS = (50, 100)
DOMAINS = ("MIT", "XJTU", "TJU", "NASA", "CALCE", "HUST")
HORIZONS = (125, 250, 500, 750, 1000, 1500, 2000)
BUDGETS = (0.10, 0.20, 0.40)
ALPHA = 0.05
SCALES = (0.3, 0.6, 1.2, 2.0)
SEEDS = tuple(range(20))
MODEL_FEATURES = ("q0_Ah", "soh_last", "slope_1e3", "drop10", "drop_half",
                  "std_soh", "n_regen", "regen_max")


def finite_quantile_min_labels(alpha: float) -> int:
    n = 1
    while math.ceil((n + 1) * (1 - alpha)) > n:
        n += 1
    return n


def main() -> None:
    source = SOURCE / "cycle_source_verified_v3.csv"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    provenance = json.loads((SOURCE / "source_v3_provenance.json").read_text(encoding="utf-8"))
    if digest != provenance["output_sha256"]:
        raise AssertionError("Source table differs from the six-domain audit manifest")
    rows_by_landmark = {}
    input_hashes = {"cycle_source_verified_v3.csv": digest}
    for landmark in LANDMARKS:
        name = f"landmark_{landmark}_features.csv"
        path = SOURCE / name
        input_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if any(float(r["followup_cycle"]) <= landmark for r in rows):
            raise AssertionError("Landmark cohort contains non-positive residual follow-up")
        rows_by_landmark[landmark] = rows

    min_labels = finite_quantile_min_labels(ALPHA)
    records = []
    for landmark in LANDMARKS:
        groups = defaultdict(list)
        for row in rows_by_landmark[landmark]:
            groups[row["dataset"]].append(row)
        if set(groups) != set(DOMAINS):
            raise AssertionError("Not all six domains appear in the landmark table")
        for domain in DOMAINS:
            n = len(groups[domain])
            events = sum(int(r["event_observed"]) for r in groups[domain])
            for fraction in BUDGETS:
                m = math.ceil(fraction * n)
                possible = m >= min_labels and events >= min_labels
                records.append({
                    "landmark": landmark, "domain": domain,
                    "requested_fraction": fraction, "n_target": n, "n_events": events,
                    "n_calibration": m, "actual_fraction": m / n,
                    "minimum_event_residuals_for_finite_quantile": min_labels,
                    "finite_quantile_possible_for_some_draw": int(possible),
                    "probability_of_enough_events_random_draw": float(hypergeom.sf(min_labels - 1, n, events, m)) if possible else 0.0,
                    "interpretation": "count feasibility for event-only absolute residual quantile; not all-cell coverage",
                })
    with (HERE / "calibration_budget_feasibility.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    protocol = {
        "status": "exploratory revision protocol fixed before v4 model fitting; not historically preregistered",
        "unit": "cell", "primary_landmark": 100, "secondary_landmark": 50,
        "domains": DOMAINS, "landmarks": LANDMARKS, "horizons_absolute_cycle": HORIZONS,
        "endpoint": "first of three consecutive valid observed SOH <= 0.8",
        "event_time_model": "residual time R=T-L in the cohort event-free at landmark L",
        "source_selection": "inner leave-one-source-out among the five source domains only",
        "model_features": MODEL_FEATURES,
        "feature_revision_reason": "drop_landmark equals 1-soh_last exactly; removed the redundant column before final v4 fit",
        "target_budget_rule": "ceil(p*n) with no minimum four-cell override",
        "target_budget_fractions": BUDGETS,
        "target_sampling_main": "simple random cells without replacement; no event stratification",
        "target_sampling_sensitivity": "event-stratified retrospective draws, separately labelled",
        "shift_prior_candidates": (0.25, 0.5, 1.0),
        "shift_prior_selection": "source-only inner validation with random 20% source calibration draws and censored NLL",
        "scale_candidates": SCALES, "training_seeds": SEEDS,
        "conformal_alpha": ALPHA, "min_event_residuals_for_finite_quantile": min_labels,
        "conformal_scope": "event-residual count feasibility only, not censor-aware all-cell coverage",
        "input_sha256": input_hashes,
        "legacy_v3": "eda/rebuild_v2 remains unchanged as a historical audit",
    }
    (HERE / "protocol_v4.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in records:
        if row["landmark"] == 100 and row["requested_fraction"] == 0.20:
            print(f"{row['domain']}: {row['n_calibration']}/{row['n_target']} cal; "
                  f"P(>= {min_labels} event residuals)={row['probability_of_enough_events_random_draw']:.6g}")


if __name__ == "__main__":
    main()
