"""Repeat source-selected residual-time XGBoost AFT under seeds 0..19."""
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
SEEDS = tuple(range(20))


def main() -> None:
    protocol = json.loads((HERE / "protocol_v4.json").read_text(encoding="utf-8"))
    if tuple(protocol["training_seeds"]) != SEEDS:
        raise AssertionError("Seed list differs from the v4 protocol")
    with (HERE / "benchmark_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        selected = {(int(r["landmark"]), r["held_out_domain"]): float(r["selected_candidate"])
                    for r in csv.DictReader(stream) if r["model"] == "xgb_aft"}
    with (HERE / "benchmark_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        seed0 = {(int(r["landmark"]), r["held_out_domain"], r["cell_id"]): float(r["aft_mu"])
                 for r in csv.DictReader(stream) if r["model"] == "xgb_aft"}
    predictions, metrics = [], []
    for landmark in protocol["landmarks"]:
        data = base.load(landmark)
        x, time, event, domain = (data[k] for k in ("x", "time", "event", "domain"))
        for held in protocol["domains"]:
            train, test = domain != held, domain == held
            target_rows = [data["rows"][i] for i in np.flatnonzero(test)]
            for seed in SEEDS:
                pred = base.fit_predict("xgb_aft", selected[(landmark, held)],
                                        x[train], x[test], base.structured(event[train], time[train]), seed=seed)
                mu = np.asarray(pred["mu"], dtype=float)
                if seed == 0:
                    reference = np.array([seed0[(landmark, held, r["cell_id"])] for r in target_rows])
                    if not np.allclose(mu, reference, atol=1e-6, rtol=1e-6):
                        raise AssertionError(f"Seed-zero fit differs from benchmark: {landmark}/{held}")
                residual_median = np.exp(mu)
                e, t = event[test], time[test]
                event_mape = np.abs(landmark + residual_median[e] - (landmark + t[e])) / (landmark + t[e]) * 100
                cindex = float(concordance_index_censored(e, t, -mu)[0])
                usable = []
                for horizon in base.HORIZONS:
                    if horizon <= landmark:
                        continue
                    rh = horizon - landmark
                    risk = 1 - base.survival_at(pred, rh)
                    brier, _, _ = base.legacy.ipcw_brier(t, e, risk, rh)
                    if brier is not None:
                        usable.append((horizon, brier))
                ibs = None
                if len(usable) >= 2:
                    grid, values = (np.asarray(z, dtype=float) for z in zip(*usable))
                    ibs = float(np.trapezoid(values, grid) / (grid[-1] - grid[0]))
                metrics.append({"landmark": landmark, "held_out_domain": held, "seed": seed,
                                "aft_scale": pred["sigma"], "n_target": len(target_rows),
                                "n_events": int(e.sum()), "event_medape_pct": float(np.median(event_mape)),
                                "harrell_c": cindex, "grid_integrated_brier": ibs,
                                "n_supported_horizons": len(usable)})
                for i, row in enumerate(target_rows):
                    predictions.append({"landmark": landmark, "held_out_domain": held, "seed": seed,
                                        "cell_id": row["cell_id"], "event_observed": int(e[i]),
                                        "residual_followup": float(t[i]),
                                        "aft_mu": float(mu[i]), "aft_sigma": pred["sigma"],
                                        "median_total_cycle": float(landmark + residual_median[i])})
            domain_metrics = [r for r in metrics if r["landmark"] == landmark and r["held_out_domain"] == held]
            print(f"L={landmark} {held}: seed MedAPE {min(r['event_medape_pct'] for r in domain_metrics):.2f}.."
                  f"{max(r['event_medape_pct'] for r in domain_metrics):.2f}", flush=True)
    for filename, data in (("seed_predictions.csv", predictions), ("seed_metrics.csv", metrics)):
        with (HERE / filename).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    output = {}
    for landmark in protocol["landmarks"]:
        output[str(landmark)] = {}
        for domain in protocol["domains"]:
            fold = [r for r in metrics if r["landmark"] == landmark and r["held_out_domain"] == domain]
            output[str(landmark)][domain] = {
                "seeds": list(SEEDS), "aft_scale": fold[0]["aft_scale"],
                "medape_min_max": [min(r["event_medape_pct"] for r in fold),
                                   max(r["event_medape_pct"] for r in fold)],
                "cindex_min_max": [min(r["harrell_c"] for r in fold),
                                   max(r["harrell_c"] for r in fold)],
                "ibs_min_max": [min(r["grid_integrated_brier"] for r in fold if r["grid_integrated_brier"] is not None),
                                max(r["grid_integrated_brier"] for r in fold if r["grid_integrated_brier"] is not None)]
                if any(r["grid_integrated_brier"] is not None for r in fold) else None,
                "note": "fixed outer domain and source-selected configuration; seeds measure fitting variation, not test-sample CI",
            }
    (HERE / "seed_sensitivity.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
