"""Cross the 20 AFT training seeds with the fixed 20% target draw memberships."""
from __future__ import annotations

import csv
import importlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

recal = importlib.import_module("02_target_recalibration")
HERE = Path(__file__).resolve().parent


def load_csv(name: str) -> list[dict]:
    with (HERE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    protocol = json.loads((HERE / "protocol_v4.json").read_text(encoding="utf-8"))
    selected_prior = json.loads((HERE / "source_prior_selection.json").read_text(encoding="utf-8"))["selected_prior_log_sd"]
    predictions = defaultdict(list)
    for r in load_csv("seed_predictions.csv"):
        predictions[(int(r["landmark"]), r["held_out_domain"], int(r["seed"]))].append(r)
    membership = defaultdict(dict)
    for r in load_csv("recalibration_membership.csv"):
        if abs(float(r["budget_fraction"]) - 0.20) < 1e-12:
            membership[(int(r["landmark"]), r["domain"], int(r["split"]))][r["cell_id"]] = r["role"]
    records = []
    for landmark in protocol["landmarks"]:
        for domain in protocol["domains"]:
            prior_sd = float(selected_prior[str(landmark)][domain])
            for seed in protocol["training_seeds"]:
                fold = sorted(predictions[(landmark, domain, seed)], key=lambda r: r["cell_id"])
                ids = [r["cell_id"] for r in fold]
                time = np.array([float(r["residual_followup"]) for r in fold])
                event = np.array([r["event_observed"] == "1" for r in fold])
                mu = np.array([float(r["aft_mu"]) for r in fold])
                sigma_values = {float(r["aft_sigma"]) for r in fold}
                if len(sigma_values) != 1:
                    raise AssertionError("Seed fold uses multiple scales")
                sigma = sigma_values.pop()
                for split in range(30):
                    roles = membership[(landmark, domain, split)]
                    if set(roles) != set(ids):
                        raise AssertionError("Seed and calibration folds contain different cells")
                    cal = np.array([i for i,k in enumerate(ids) if roles[k] == "calibration"])
                    test = np.array([i for i,k in enumerate(ids) if roles[k] == "evaluation"])
                    shift = recal.fit_shift_with_prior(time[cal], event[cal], mu[cal], sigma, prior_sd)
                    raw_nll = float(recal.nll(time[test], event[test], mu[test], sigma).mean())
                    shifted_nll = float(recal.nll(time[test], event[test], mu[test], sigma, shift).mean())
                    usable = []
                    for h in recal.HORIZONS:
                        if h <= landmark:
                            continue
                        rh = h - landmark
                        raw = recal.risk(mu[test], sigma, rh)
                        shifted = recal.risk(mu[test], sigma, rh, shift)
                        b0, _, _ = recal.legacy.ipcw_brier(time[test], event[test], raw, rh)
                        b1, _, _ = recal.legacy.ipcw_brier(time[test], event[test], shifted, rh)
                        if (b0 is None) != (b1 is None):
                            raise AssertionError("Paired Brier support differs")
                        if b0 is not None:
                            usable.append((h,b0,b1))
                    delta = None
                    if len(usable) >= 2:
                        grid, b0, b1 = (np.asarray(z, dtype=float) for z in zip(*usable))
                        delta = float(np.trapezoid(b1-b0, grid) / (grid[-1]-grid[0]))
                    records.append({"landmark": landmark, "domain": domain, "seed": seed, "split": split,
                                    "n_calibration": len(cal), "n_calibration_events": int(event[cal].sum()),
                                    "n_evaluation": len(test), "n_evaluation_events": int(event[test].sum()),
                                    "aft_scale": sigma, "log_time_shift": shift,
                                    "prior_log_shift_sd": prior_sd,
                                    "nll_difference": shifted_nll - raw_nll,
                                    "n_supported_horizons": len(usable),
                                    "grid_brier_difference": delta if delta is not None else ""})
            selected = [r for r in records if r["landmark"] == landmark and r["domain"] == domain]
            usable = np.array([r["grid_brier_difference"] for r in selected if r["grid_brier_difference"] != ""], dtype=float)
            print(f"L={landmark} {domain}: supported={len(usable)}/600, "
                  f"improved={int((usable<0).sum()) if len(usable) else 'NA'}", flush=True)
    with (HERE / "seed_target_recalibration_scores.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    output = {}
    for landmark in protocol["landmarks"]:
        output[str(landmark)] = {}
        for domain in protocol["domains"]:
            fold = [r for r in records if r["landmark"] == landmark and r["domain"] == domain]
            supported = np.array([r["grid_brier_difference"] for r in fold if r["grid_brier_difference"] != ""], dtype=float)
            per_seed_mean = []
            for seed in protocol["training_seeds"]:
                values = [r["grid_brier_difference"] for r in fold if r["seed"] == seed and r["grid_brier_difference"] != ""]
                if values:
                    per_seed_mean.append(float(np.mean(values)))
            output[str(landmark)][domain] = {
                "prior_log_shift_sd": float(selected_prior[str(landmark)][domain]),
                "n_training_seeds": 20, "n_overlapping_target_draws_per_seed": 30,
                "n_supported_paired_brier": len(supported),
                "fraction_of_supported_seed_draw_pairs_improved": float((supported < 0).mean()) if len(supported) else None,
                "mean_paired_brier_difference": float(supported.mean()) if len(supported) else None,
                "per_seed_mean_difference_min_max": [min(per_seed_mean), max(per_seed_mean)] if per_seed_mean else None,
                "n_seed_means_improved": sum(x < 0 for x in per_seed_mean),
                "note": "overlapping target draws and common cells; descriptive sensitivity, not independent inferential replicates",
            }
    (HERE / "seed_target_recalibration_summary.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
