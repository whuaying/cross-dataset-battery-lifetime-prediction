"""Training-data correction ablation on a fixed source-certified test cohort.

The pre-five-domain-correction table is the historical comparator. Both
branches train on the same cells and use fixed source-only hyperparameters.
Every target cell is evaluated against the same v3 certified endpoint and
features, so a change in target label cannot masquerade as improvement.
"""
from __future__ import annotations

import csv
import importlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from sksurv.metrics import concordance_index_censored

base = importlib.import_module("01_residual_survival_benchmark")
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "rebuild_v2"
RAW = SOURCE / "cycle_diagnostic_corrected.csv"


def raw_cells() -> dict:
    out = defaultdict(list)
    with RAW.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            out[(row["dataset"], row["cell_id"])].append(row)
    return out


def first_run(cycle: list[int], soh: list[float]) -> int | None:
    for i in range(len(soh) - 2):
        if soh[i] <= 0.8 and soh[i+1] <= 0.8 and soh[i+2] <= 0.8:
            return cycle[i]
    return None


def feature_at_landmark(records: list[dict], landmark: int) -> dict | None:
    rec = sorted(records, key=lambda r: int(r["cycle_idx"]))
    cycle = [int(r["cycle_idx"]) for r in rec]
    if len(cycle) != len(set(cycle)) or cycle[-1] <= landmark:
        return None
    capacity = [float(r["cap_Ah"]) for r in rec]
    q0 = capacity[0]
    soh = [q / q0 for q in capacity]
    event = first_run(cycle, soh)
    if event is not None and event <= landmark:
        return None
    observed = [(c, s) for c, s in zip(cycle, soh) if c <= landmark]
    if len(observed) < (10 if landmark == 50 else 20):
        return None
    xs, ys = zip(*observed)
    xm, ym = statistics.mean(xs), statistics.mean(ys)
    denom = sum((x-xm)**2 for x in xs)
    slope = sum((x-xm)*(y-ym) for x,y in observed) / denom if denom else 0
    changes = [b-a for a,b in zip(ys, ys[1:])]
    early = [s for c,s in observed if c <= 10]
    half = [s for c,s in observed if c <= landmark//2]
    values = {
        "q0_Ah": q0, "soh_last": ys[-1], "slope_1e3": slope * 1000,
        "drop10": 1-early[-1] if early else np.nan,
        "drop_half": 1-half[-1] if half else np.nan,
        "std_soh": statistics.pstdev(ys),
        "n_regen": sum(d > 0.001 for d in changes),
        "regen_max": max(changes, default=0),
    }
    return {"features": np.array([values[k] for k in base.FEATURES], dtype=float),
            "event": event is not None, "time": float(event if event is not None else cycle[-1]) - landmark,
            "raw_first_event": event}


def main() -> None:
    cells = raw_cells()
    with (HERE / "benchmark_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        choice = {(int(r["landmark"]), r["held_out_domain"], r["model"]): float(r["selected_candidate"])
                  for r in csv.DictReader(stream) if r["model"] in {"xgb_aft", "cox"}}
    cohort_rows, score_rows, prediction_rows = [], [], []
    for landmark in (50, 100):
        clean = base.load(landmark)
        raw = [feature_at_landmark(cells[(r["dataset"], r["cell_id"])], landmark) for r in clean["rows"]]
        common = np.array([r is not None for r in raw], dtype=bool)
        domains = clean["domain"][common]
        clean_x, clean_time, clean_event = clean["x"][common], clean["time"][common], clean["event"][common]
        raw_x = np.vstack([r["features"] for r in raw if r is not None])
        raw_time = np.array([r["time"] for r in raw if r is not None])
        raw_event = np.array([r["event"] for r in raw if r is not None])
        identities = [r for r, good in zip(clean["rows"], common) if good]
        print(f"L={landmark}: certified cohort={len(clean['rows'])}, common raw/certified={int(common.sum())}", flush=True)
        for held in base.DOMAINS:
            tr, te = domains != held, domains == held
            cohort_rows.append({"landmark": landmark, "held_out_domain": held,
                                "n_certified_cohort": int((clean["domain"] == held).sum()),
                                "n_common_cohort": int(te.sum()),
                                "n_excluded_by_raw_eligibility": int((clean["domain"] == held).sum() - te.sum()),
                                "n_changed_training_event_status": int((raw_event[tr] != clean_event[tr]).sum()),
                                "n_changed_training_event_time_or_censor": int((raw_time[tr] != clean_time[tr]).sum()),
                                "n_changed_feature_rows": int(np.any(~np.isclose(raw_x[tr], clean_x[tr], equal_nan=True), axis=1).sum())})
            for model in ("xgb_aft", "cox"):
                candidate = choice[(landmark, held, model)]
                for variant in ("before_source_correction", "certified_v3"):
                    if variant == "before_source_correction":
                        xtr, ttr, etr = raw_x[tr], raw_time[tr], raw_event[tr]
                    else:
                        xtr, ttr, etr = clean_x[tr], clean_time[tr], clean_event[tr]
                    # Test features and truth are always the source-certified version.
                    pred = base.fit_predict(model, candidate, xtr, clean_x[te],
                                            base.structured(etr, ttr))
                    if not pred["fitted"]:
                        score_rows.append({"landmark": landmark, "held_out_domain": held,
                                           "model": model, "training_variant": variant,
                                           "n_target_common": int(te.sum()), "n_target_events": int(clean_event[te].sum()),
                                           "fit_converged": 0, "harrell_c": "", "event_medape_pct": "",
                                           "grid_integrated_brier": "", "n_supported_horizons": ""})
                        continue
                    med = base.median_residual(pred)
                    evmed, evt = med[clean_event[te]], clean_time[te][clean_event[te]] + landmark
                    available = np.isfinite(evmed)
                    medape = float(np.median(abs(landmark + evmed[available] - evt[available]) / evt[available] * 100)) if available.any() else None
                    c = float(concordance_index_censored(clean_event[te], clean_time[te], np.asarray(pred["risk_score"]))[0])
                    usable = []
                    for h in base.HORIZONS:
                        if h <= landmark:
                            continue
                        rh = h-landmark
                        brier, _, _ = base.legacy.ipcw_brier(clean_time[te], clean_event[te],
                                                               1 - base.survival_at(pred, rh), rh)
                        if pred["kind"] == "step" and rh >= pred["source_max_followup"]:
                            brier = None
                        if brier is not None:
                            usable.append((h,brier))
                    ibs = None
                    if len(usable) >= 2:
                        grid, score = (np.array(z, dtype=float) for z in zip(*usable))
                        ibs = float(np.trapezoid(score, grid) / (grid[-1] - grid[0]))
                    score_rows.append({"landmark": landmark, "held_out_domain": held,
                                       "model": model, "training_variant": variant,
                                       "n_target_common": int(te.sum()), "n_target_events": int(clean_event[te].sum()),
                                       "fit_converged": 1, "harrell_c": c,
                                       "event_medape_pct": medape, "grid_integrated_brier": ibs,
                                       "n_supported_horizons": len(usable)})
                    for i, idx in enumerate(np.flatnonzero(te)):
                        prediction_rows.append({"landmark": landmark, "held_out_domain": held,
                                                "cell_id": identities[idx]["cell_id"], "model": model,
                                                "training_variant": variant,
                                                "certified_event": int(clean_event[idx]),
                                                "certified_followup": float(clean_time[idx] + landmark),
                                                "median_total_cycle": float(med[i] + landmark) if np.isfinite(med[i]) else ""})
    for name, data in (("source_correction_cohort.csv", cohort_rows),
                       ("source_correction_metrics.csv", score_rows),
                       ("source_correction_predictions.csv", prediction_rows)):
        with (HERE / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (HERE / "source_correction_protocol.json").write_text(json.dumps({
        "historical_training_input": str(RAW),
        "certified_training_input": str(SOURCE / "cycle_source_verified_v3.csv"),
        "target_reference": "fixed certified-v3 early features and three-record EOL on the common eligible cells",
        "model_configuration": "source-selected from the v4 certified three-record benchmark, fixed in both branches",
        "interpretation": "paired training-data correction audit; excluded-cell composition reported separately; not a causal decomposition",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
