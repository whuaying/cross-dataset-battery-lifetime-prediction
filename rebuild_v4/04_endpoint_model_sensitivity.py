"""Rerun two survival models for 1/3/5-record EOL on common landmark cells."""
from __future__ import annotations

import csv
import importlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sksurv.metrics import concordance_index_censored

base = importlib.import_module("01_residual_survival_benchmark")
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "rebuild_v2"
ENDPOINT = {1: "raw_first_crossing", 3: "sustained_3", 5: "sustained_5"}


def rows(name: str) -> list[dict]:
    with (SOURCE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    eol = {(r["dataset"], r["cell_id"]): r for r in rows("eol_sensitivity_cells.csv")}
    audit = {(r["dataset"], r["cell_id"]): r for r in rows("cell_audit.csv")}
    with (HERE / "benchmark_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        selected = {(int(r["landmark"]), r["held_out_domain"], r["model"]): float(r["selected_candidate"])
                    for r in csv.DictReader(stream) if r["model"] in {"xgb_aft", "cox"}}
    cells, metrics, counts, horizon_rows = [], [], [], []
    for landmark in (50, 100):
        original = base.load(landmark)
        keep = np.array([not eol[(r["dataset"], r["cell_id"])]["raw_first_crossing"] or
                         float(eol[(r["dataset"], r["cell_id"])]["raw_first_crossing"]) > landmark
                         for r in original["rows"]], dtype=bool)
        features = original["x"][keep]
        cohort = [r for r, good in zip(original["rows"], keep) if good]
        domains = np.array([r["dataset"] for r in cohort])
        if not len(cohort):
            raise AssertionError("Common endpoint cohort empty")
        print(f"L={landmark}: main cohort={len(original['rows'])}, common={len(cohort)}", flush=True)
        for rule, name in ENDPOINT.items():
            event = np.array([bool(eol[(r["dataset"], r["cell_id"])][name]) for r in cohort])
            followup = np.array([float(eol[(r["dataset"], r["cell_id"])][name])
                                 if eol[(r["dataset"], r["cell_id"])][name]
                                 else float(audit[(r["dataset"], r["cell_id"])]["last_cycle"])
                                 for r in cohort])
            residual = followup - landmark
            if not np.all(residual > 0):
                raise AssertionError("Common cohort contains event/censor before landmark")
            for domain in base.DOMAINS:
                tr, te = domains != domain, domains == domain
                counts.append({"landmark": landmark, "domain": domain, "endpoint_records": rule,
                               "n_common_cells": int(te.sum()), "n_events": int(event[te].sum()),
                               "n_censored": int((~event[te]).sum())})
                for model in ("xgb_aft", "cox"):
                    candidate = selected[(landmark, domain, model)]
                    pred = base.fit_predict(model, candidate, features[tr], features[te],
                                            base.structured(event[tr], residual[tr]))
                    if not pred["fitted"]:
                        raise RuntimeError(f"Non-converged endpoint fit: {landmark}/{domain}/{rule}/{model}")
                    median = base.median_residual(pred)
                    event_median = median[event[te]]
                    observed_total = followup[te][event[te]]
                    available = np.isfinite(event_median)
                    medape = float(np.median(np.abs(landmark + event_median[available] - observed_total[available]) /
                                             observed_total[available] * 100)) if available.any() else None
                    c = float(concordance_index_censored(event[te], residual[te],
                                                         np.asarray(pred["risk_score"]))[0])
                    usable = []
                    for h in base.HORIZONS:
                        if h <= landmark:
                            continue
                        rh = h - landmark
                        risk = 1 - base.survival_at(pred, rh)
                        brier, _, _ = base.legacy.ipcw_brier(residual[te], event[te], risk, rh)
                        if pred["kind"] == "step" and rh >= pred["source_max_followup"]:
                            brier = None
                        horizon_rows.append({"landmark": landmark, "domain": domain,
                                             "endpoint_records": rule, "model": model,
                                             "horizon_cycle": h, "ipcw_brier": brier})
                        if brier is not None:
                            usable.append((h, brier))
                    ibs = None
                    if len(usable) >= 2:
                        grid, values = (np.array(z, dtype=float) for z in zip(*usable))
                        ibs = float(np.trapezoid(values, grid) / (grid[-1] - grid[0]))
                    metrics.append({"landmark": landmark, "domain": domain, "endpoint_records": rule,
                                    "model": model, "n_common_cells": int(te.sum()),
                                    "n_events": int(event[te].sum()), "event_medape_pct": medape,
                                    "n_event_medians_available": int(available.sum()),
                                    "harrell_c": c, "n_supported_horizons": len(usable),
                                    "grid_integrated_brier": ibs,
                                    "note": "different endpoint rules define different outcomes; compare stability, not accuracy improvement"})
                    for i, idx in enumerate(np.flatnonzero(te)):
                        cells.append({"landmark": landmark, "domain": domain,
                                      "cell_id": cohort[idx]["cell_id"], "endpoint_records": rule,
                                      "model": model, "event_observed": int(event[idx]),
                                      "followup_cycle": float(followup[idx]),
                                      "median_total_cycle": float(landmark + median[i]) if np.isfinite(median[i]) else ""})
    # Endpoint rules can change the censoring support. Compare their integrated
    # scores only over horizons supported under all three rules for this fold.
    for metric in metrics:
        matching = [r for r in horizon_rows if r["landmark"] == metric["landmark"] and
                    r["domain"] == metric["domain"] and r["model"] == metric["model"]]
        shared = sorted(h for h in base.HORIZONS if h > metric["landmark"] and
                        all(any(r["endpoint_records"] == rule and r["horizon_cycle"] == h and
                                r["ipcw_brier"] is not None for r in matching)
                            for rule in ENDPOINT))
        own = [r for r in matching if r["endpoint_records"] == metric["endpoint_records"] and
               r["horizon_cycle"] in shared]
        own.sort(key=lambda r: r["horizon_cycle"])
        metric["n_common_supported_horizons"] = len(shared)
        metric["common_grid_first_cycle"] = shared[0] if shared else ""
        metric["common_grid_last_cycle"] = shared[-1] if shared else ""
        metric["common_grid_integrated_brier"] = (float(np.trapezoid(
            [r["ipcw_brier"] for r in own], shared) / (shared[-1] - shared[0]))
            if len(shared) >= 2 else None)
    for filename, data in (("endpoint_common_cohort.csv", counts),
                           ("endpoint_model_metrics.csv", metrics),
                           ("endpoint_model_predictions.csv", cells),
                           ("endpoint_model_horizons.csv", horizon_rows)):
        with (HERE / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (HERE / "endpoint_sensitivity_protocol.json").write_text(json.dumps({
        "input": "v3 endpoint labels and cycle-50/100 features",
        "common_cohort": "v3 landmark-eligible cells with raw one-record EOL after landmark",
        "models": ["xgb_aft", "cox"],
        "hyperparameters": "source-only configurations selected for the three-record v4 benchmark, frozen across endpoint variants",
        "endpoint_rules": ENDPOINT,
        "interpretation": "different rules define different outcomes; model ranking and support sensitivity only",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
