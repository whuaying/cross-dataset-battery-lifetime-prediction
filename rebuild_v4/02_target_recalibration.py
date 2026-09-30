"""Recalibrate residual-life probabilities on disjoint random target cells.

This is a frozen-model, seed-0 exploratory analysis. It does not estimate
training-seed variation or conformal coverage; these remain separate gates.
"""
from __future__ import annotations

import csv
import importlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rebuild_v2"))
legacy = importlib.import_module("03e_survival_calibration")
HORIZONS = (125, 250, 500, 750, 1000, 1500, 2000)
PRIOR_LOG_SD = 0.5
SHIFT_BOUND = float(np.log(4))
N_SPLITS = 30


def nll(time: np.ndarray, event: np.ndarray, mu: np.ndarray, sigma: float,
        shift: float = 0) -> np.ndarray:
    z = (np.log(time) - mu - shift) / sigma
    return np.where(event,
                    np.log(time * sigma * np.sqrt(2 * np.pi)) + 0.5 * z**2,
                    -log_ndtr(-z))


def risk(mu: np.ndarray, sigma: float, time: float, shift: float = 0) -> np.ndarray:
    return 1 - np.exp(log_ndtr((mu + shift - np.log(time)) / sigma))


def fit_shift(time: np.ndarray, event: np.ndarray, mu: np.ndarray, sigma: float) -> float:
    return fit_shift_with_prior(time, event, mu, sigma, PRIOR_LOG_SD)


def fit_shift_with_prior(time: np.ndarray, event: np.ndarray, mu: np.ndarray,
                         sigma: float, prior_log_sd: float) -> float:
    objective = lambda d: float(nll(time, event, mu, sigma, d).sum()
                                + d**2 / (2 * prior_log_sd**2))
    result = minimize_scalar(objective, method="bounded", bounds=(-SHIFT_BOUND, SHIFT_BOUND),
                             options={"xatol": 1e-8})
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError("Target shift optimization failed")
    return float(result.x)


def summary(values: list[float]) -> dict | None:
    if not values:
        return None
    x = np.asarray(values, dtype=float)
    return {"n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
            "minimum": float(x.min()), "maximum": float(x.max()),
            "n_improved": int((x < -1e-10).sum()), "n_worsened": int((x > 1e-10).sum())}


def main() -> None:
    protocol = json.loads((HERE / "protocol_v4.json").read_text(encoding="utf-8"))
    selected_prior = json.loads((HERE / "source_prior_selection.json").read_text(encoding="utf-8"))["selected_prior_log_sd"]
    with (HERE / "benchmark_predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        grouped = defaultdict(list)
        for row in csv.DictReader(stream):
            if row["model"] == "xgb_aft":
                grouped[(int(row["landmark"]), row["held_out_domain"])].append(row)
    membership, scores, horizons = [], [], []
    result = {"protocol": {"model": "v4 source-selected residual-life XGBoost AFT seed 0",
                           "draws": N_SPLITS, "sampling": "simple random without event stratification",
                           "budgets": protocol["target_budget_fractions"],
                           "prior_log_shift_sd": "source-only selected per landmark and held-out domain",
                           "prior_selection_file": "source_prior_selection.json",
                           "log_shift_bound": SHIFT_BOUND,
                           "note": "repeated draws overlap; one seed; exploratory target adaptation"},
              "landmarks": {}}
    for landmark in protocol["landmarks"]:
        result["landmarks"][str(landmark)] = {}
        for domain in protocol["domains"]:
            prior_sd = float(selected_prior[str(landmark)][domain])
            rows = sorted(grouped[(landmark, domain)], key=lambda r: r["cell_id"])
            if len(rows) == 0:
                raise AssertionError("Missing benchmark predictions")
            time = np.array([float(r["residual_followup"]) for r in rows])
            event = np.array([r["event_observed"] == "1" for r in rows])
            mu = np.array([float(r["aft_mu"]) for r in rows])
            sigma_values = {float(r["aft_sigma"]) for r in rows}
            if len(sigma_values) != 1:
                raise AssertionError("A source fold has inconsistent AFT scale")
            sigma = sigma_values.pop()
            for fraction in protocol["target_budget_fractions"]:
                m = math.ceil(fraction * len(rows))
                key = f"{fraction:.2f}"
                for split in range(N_SPLITS):
                    rng = np.random.default_rng(20261001 + landmark * 10000 +
                                                protocol["domains"].index(domain) * 1000 +
                                                int(100 * fraction) * 100 + split)
                    perm = rng.permutation(len(rows))
                    cal, test = perm[:m], perm[m:]
                    assert len(set(cal) & set(test)) == 0 and len(test) > 0
                    shift = fit_shift_with_prior(time[cal], event[cal], mu[cal], sigma, prior_sd)
                    for role, indexes in (("calibration", cal), ("evaluation", test)):
                        for i in indexes:
                            membership.append({"landmark": landmark, "domain": domain,
                                               "budget_fraction": fraction, "split": split,
                                               "cell_id": rows[i]["cell_id"], "role": role,
                                               "event_observed": int(event[i]),
                                               "residual_followup": float(time[i]),
                                               "baseline_mu": float(mu[i]), "aft_sigma": sigma})
                    raw_nll = float(nll(time[test], event[test], mu[test], sigma).mean())
                    new_nll = float(nll(time[test], event[test], mu[test], sigma, shift).mean())
                    usable = []
                    for horizon in HORIZONS:
                        if horizon <= landmark:
                            continue
                        residual_horizon = horizon - landmark
                        r0, r1 = risk(mu[test], sigma, residual_horizon), risk(mu[test], sigma, residual_horizon, shift)
                        b0, g, n_late = legacy.ipcw_brier(time[test], event[test], r0, residual_horizon)
                        b1, _, _ = legacy.ipcw_brier(time[test], event[test], r1, residual_horizon)
                        if (b0 is None) != (b1 is None):
                            raise AssertionError("Paired Brier support differs")
                        horizons.append({"landmark": landmark, "domain": domain,
                                         "budget_fraction": fraction, "split": split,
                                         "horizon_cycle": horizon, "n_evaluation": len(test),
                                         "n_late": n_late, "censor_survival": g,
                                         "supported": int(b0 is not None),
                                         "raw_brier": b0, "shifted_brier": b1,
                                         "brier_difference": b1 - b0 if b0 is not None else "",
                                         "observed_km_risk": 1 - legacy.km_survival(time[test], event[test], residual_horizon) if b0 is not None else "",
                                         "raw_mean_risk": float(r0.mean()),
                                         "shifted_mean_risk": float(r1.mean())})
                        if b0 is not None:
                            usable.append((horizon, b0, b1))
                    raw_ibs = new_ibs = None
                    if len(usable) >= 2:
                        grid, raw, shifted = (np.array(z, dtype=float) for z in zip(*usable))
                        raw_ibs = float(np.trapezoid(raw, grid) / (grid[-1] - grid[0]))
                        new_ibs = float(np.trapezoid(shifted, grid) / (grid[-1] - grid[0]))
                    scores.append({"landmark": landmark, "domain": domain,
                                   "budget_fraction": fraction, "split": split,
                                   "n_calibration": len(cal), "n_calibration_events": int(event[cal].sum()),
                                   "prior_log_shift_sd": prior_sd,
                                   "n_evaluation": len(test), "n_evaluation_events": int(event[test].sum()),
                                   "shift": shift, "median_multiplier": float(np.exp(shift)),
                                   "raw_mean_nll": raw_nll, "shifted_mean_nll": new_nll,
                                   "nll_difference": new_nll - raw_nll,
                                   "n_supported_horizons": len(usable),
                                   "first_supported_horizon": usable[0][0] if len(usable) >= 2 else "",
                                   "last_supported_horizon": usable[-1][0] if len(usable) >= 2 else "",
                                   "raw_grid_brier": raw_ibs, "shifted_grid_brier": new_ibs,
                                   "grid_brier_difference": new_ibs - raw_ibs if raw_ibs is not None else ""})
                selected = [r for r in scores if r["landmark"] == landmark and
                            r["domain"] == domain and r["budget_fraction"] == fraction]
                brier_delta = [r["grid_brier_difference"] for r in selected if r["grid_brier_difference"] != ""]
                result["landmarks"][str(landmark)].setdefault(domain, {})[key] = {
                    "n_total": len(rows), "n_calibration": m,
                    "prior_log_shift_sd": prior_sd,
                    "n_calibration_events_range": [min(r["n_calibration_events"] for r in selected),
                                                   max(r["n_calibration_events"] for r in selected)],
                    "n_zero_event_calibrations": sum(r["n_calibration_events"] == 0 for r in selected),
                    "paired_integrated_brier_difference": summary(brier_delta),
                    "paired_nll_difference": summary([r["nll_difference"] for r in selected]),
                    "n_integrated_brier_unavailable": N_SPLITS - len(brier_delta),
                }
            x = result["landmarks"][str(landmark)][domain]["0.20"]
            print(f"L={landmark} {domain}: 20%={x['n_calibration']}/{x['n_total']}, "
                  f"IBS={x['paired_integrated_brier_difference']}, zero-event={x['n_zero_event_calibrations']}", flush=True)
    for name, data in (("recalibration_membership.csv", membership),
                       ("recalibration_scores.csv", scores),
                       ("recalibration_horizons.csv", horizons)):
        with (HERE / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (HERE / "recalibration_budget_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
