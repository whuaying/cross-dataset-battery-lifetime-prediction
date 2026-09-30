"""Source-held-out residual-life benchmark with source-only model selection.

The v3 cohort and early features are frozen. All model labels are R=T-L for
events and R>C-L for right-censored cells. Target outcomes score predictions
only; no target outcome enters imputation, model fitting, or tuning.
"""
from __future__ import annotations

import csv
import importlib
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import xgboost as xgb
from scipy.optimize import minimize
from scipy.special import log_ndtr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.nonparametric import kaplan_meier_estimator

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "rebuild_v2"
sys.path.insert(0, str(SOURCE))
legacy = importlib.import_module("03e_survival_calibration")
FEATURES = ("q0_Ah", "soh_last", "slope_1e3", "drop10", "drop_half",
            "std_soh", "n_regen", "regen_max")
DOMAINS = ("MIT", "XJTU", "TJU", "NASA", "CALCE", "HUST")
HORIZONS = (125, 250, 500, 750, 1000, 1500, 2000)
CANDIDATES = {"xgb_aft": (0.3, 0.6, 1.2, 2.0),
              "lognormal_aft": (0.1, 1.0, 10.0),
              "cox": (0.1, 1.0, 10.0),
              "rsf": (3, 8)}
N_BOOST = 300
N_TREES = 100


def load(landmark: int) -> dict:
    with (SOURCE / f"landmark_{landmark}_features.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    x = np.array([[float(r[k]) if r[k] else np.nan for k in FEATURES] for r in rows], dtype=float)
    time = np.array([float(r["followup_cycle"]) - landmark for r in rows])
    event = np.array([r["event_observed"] == "1" for r in rows])
    domain = np.array([r["dataset"] for r in rows])
    if not np.all(time > 0) or not np.all(np.isfinite(x[~np.isnan(x)])):
        raise AssertionError("Residual time or feature invalid")
    if len({(r["dataset"], r["cell_id"]) for r in rows}) != len(rows):
        raise AssertionError("Duplicate cell")
    return {"rows": rows, "x": x, "time": time, "event": event, "domain": domain}


def structured(event: np.ndarray, time: np.ndarray) -> np.ndarray:
    out = np.empty(len(time), dtype=[("event", "?"), ("time", "<f8")])
    out["event"] = event
    out["time"] = time
    return out


def lognormal_nll(time: np.ndarray, event: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    z = (np.log(time) - mu) / sigma
    return np.where(event,
                    np.log(time * sigma * np.sqrt(2 * np.pi)) + 0.5 * z**2,
                    -log_ndtr(-z))


def fit_lognormal(x: np.ndarray, event: np.ndarray, time: np.ndarray, ridge: float):
    initial = np.r_[np.log(np.median(time[event])) if event.any() else np.log(np.median(time)),
                    np.zeros(x.shape[1]), np.log(1.0)]
    def objective(theta):
        mu = theta[0] + x @ theta[1:-1]
        return float(lognormal_nll(time, event, mu, np.exp(theta[-1])).sum()
                     + ridge * np.sum(theta[1:-1] ** 2) / 2)
    bounds = [(None, None)] * (len(initial) - 1) + [(math.log(0.2), math.log(3.0))]
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-9})
    if not np.isfinite(result.fun):
        raise RuntimeError("Non-finite lognormal fit")
    return result.x, bool(result.success)


def fit_predict(model_name: str, candidate: float | int | None, xtr_raw: np.ndarray,
                xte_raw: np.ndarray, ytr: np.ndarray, seed: int = 0) -> dict:
    event = ytr["event"]
    time = ytr["time"]
    if model_name == "km":
        grid, survival = kaplan_meier_estimator(event, time)
        return {"kind": "step", "grid": grid, "survival": np.tile(survival, (len(xte_raw), 1)),
                "risk_score": np.zeros(len(xte_raw)), "sigma": None,
                "fitted": True, "source_max_followup": float(time.max())}
    imputer = SimpleImputer(strategy="median").fit(xtr_raw)
    xtr = imputer.transform(xtr_raw)
    xte = imputer.transform(xte_raw)
    if model_name == "xgb_aft":
        dtrain = xgb.DMatrix(xtr, feature_names=list(FEATURES))
        dtrain.set_float_info("label_lower_bound", time)
        dtrain.set_float_info("label_upper_bound", np.where(event, time, np.inf))
        params = {"objective": "survival:aft", "eval_metric": "aft-nloglik",
                  "aft_loss_distribution": "normal", "aft_loss_distribution_scale": float(candidate),
                  "tree_method": "hist", "max_depth": 4, "eta": 0.05, "subsample": 0.8,
                  "colsample_bytree": 0.8, "lambda": 1.0, "seed": seed, "nthread": 4}
        model = xgb.train(params, dtrain, num_boost_round=N_BOOST, verbose_eval=False)
        mu = model.predict(xgb.DMatrix(xte, feature_names=list(FEATURES)), output_margin=True)
        return {"kind": "lognormal", "mu": mu, "sigma": float(candidate),
                "risk_score": -mu, "fitted": True, "source_max_followup": float(time.max())}
    scaler = StandardScaler().fit(xtr)
    xtr, xte = scaler.transform(xtr), scaler.transform(xte)
    if model_name == "lognormal_aft":
        theta, converged = fit_lognormal(xtr, event, time, float(candidate))
        mu = theta[0] + xte @ theta[1:-1]
        return {"kind": "lognormal", "mu": mu, "sigma": float(np.exp(theta[-1])),
                "risk_score": -mu, "fitted": converged,
                "source_max_followup": float(time.max())}
    if model_name == "cox":
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = CoxPHSurvivalAnalysis(alpha=float(candidate), n_iter=1000).fit(xtr, ytr)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        curves = model.predict_survival_function(xte)
        grid = np.asarray(model.unique_times_, dtype=float)
        values = np.asarray([np.asarray([curve(t) for t in grid]) for curve in curves])
        return {"kind": "step", "grid": grid, "survival": values,
                "risk_score": model.predict(xte), "sigma": None,
                "fitted": converged, "source_max_followup": float(time.max())}
    if model_name == "rsf":
        model = RandomSurvivalForest(n_estimators=N_TREES, min_samples_split=6,
                                     min_samples_leaf=int(candidate), max_features="sqrt",
                                     random_state=seed, n_jobs=4).fit(xtr, ytr)
        return {"kind": "step", "grid": np.asarray(model.unique_times_, dtype=float),
                "survival": model.predict_survival_function(xte, return_array=True),
                "risk_score": model.predict(xte), "sigma": None,
                "fitted": True, "source_max_followup": float(time.max())}
    raise ValueError(model_name)


def survival_at(pred: dict, residual_horizon: float) -> np.ndarray:
    if residual_horizon <= 0:
        return np.ones(len(pred["risk_score"]))
    if pred["kind"] == "lognormal":
        return np.exp(log_ndtr((pred["mu"] - np.log(residual_horizon)) / pred["sigma"]))
    index = int(np.searchsorted(pred["grid"], residual_horizon, side="right") - 1)
    if index < 0:
        return np.ones(len(pred["risk_score"]))
    return np.asarray(pred["survival"][:, index], dtype=float)


def median_residual(pred: dict) -> np.ndarray:
    if pred["kind"] == "lognormal":
        return np.exp(pred["mu"])
    values = np.asarray(pred["survival"])
    result = np.full(len(values), np.nan)
    for i, curve in enumerate(values):
        hit = np.flatnonzero(curve <= 0.5)
        if len(hit):
            result[i] = pred["grid"][hit[0]]
    return result


def score_validation(pred: dict, time: np.ndarray, event: np.ndarray,
                     landmark: int, model_name: str) -> float:
    if pred["kind"] == "lognormal":
        return float(lognormal_nll(time, event, pred["mu"], pred["sigma"]).mean())
    scores = []
    for horizon in HORIZONS:
        if horizon <= landmark:
            continue
        residual_horizon = horizon - landmark
        if residual_horizon >= pred["source_max_followup"]:
            continue
        brier, _, _ = legacy.ipcw_brier(time, event,
                                        1 - survival_at(pred, residual_horizon), residual_horizon)
        if brier is not None:
            scores.append(brier)
    return float(np.mean(scores)) if scores else np.nan


def main() -> None:
    protocol = json.loads((HERE / "protocol_v4.json").read_text(encoding="utf-8"))
    if tuple(protocol["scale_candidates"]) != CANDIDATES["xgb_aft"]:
        raise AssertionError("Scale candidates differ from the frozen protocol")
    if tuple(protocol["model_features"]) != FEATURES:
        raise AssertionError("Feature list differs from the frozen protocol")
    selection_rows, prediction_rows, horizon_rows, fold_rows = [], [], [], []
    for landmark in protocol["landmarks"]:
        data = load(landmark)
        x, time, event, domain = (data[k] for k in ("x", "time", "event", "domain"))
        for held in DOMAINS:
            outer_train, outer_test = domain != held, domain == held
            source_domains = [d for d in DOMAINS if d != held]
            selected = {"km": None}
            for model_name, candidates in CANDIDATES.items():
                candidate_scores = []
                for candidate in candidates:
                    inner_scores = []
                    for validation_domain in source_domains:
                        inner_train = outer_train & (domain != validation_domain)
                        inner_test = domain == validation_domain
                        ytr = structured(event[inner_train], time[inner_train])
                        pred = fit_predict(model_name, candidate, x[inner_train], x[inner_test], ytr)
                        score = score_validation(pred, time[inner_test], event[inner_test], landmark, model_name)
                        selection_rows.append({"landmark": landmark, "held_out_domain": held,
                                               "model": model_name, "candidate": candidate,
                                               "inner_validation_domain": validation_domain,
                                               "validation_score": score,
                                               "score_type": "censored NLL" if pred["kind"] == "lognormal" else "mean supported Brier",
                                               "n_validation": int(inner_test.sum()),
                                               "n_events": int(event[inner_test].sum()),
                                               "fit_converged": int(pred["fitted"])})
                        if pred["fitted"] and np.isfinite(score):
                            inner_scores.append(score)
                    candidate_scores.append(float(np.mean(inner_scores)) if len(inner_scores) == len(source_domains) else float("inf"))
                if not np.isfinite(candidate_scores).any():
                    raise RuntimeError(f"No converged source-only candidate: {landmark}/{held}/{model_name}")
                selected[model_name] = candidates[int(np.argmin(candidate_scores))]
            for model_name in ("km", *CANDIDATES):
                ytr = structured(event[outer_train], time[outer_train])
                pred = fit_predict(model_name, selected[model_name], x[outer_train], x[outer_test], ytr)
                if not pred["fitted"]:
                    raise RuntimeError(f"Final fit failed to converge: {landmark}/{held}/{model_name}")
                med = median_residual(pred)
                ttest = time[outer_test]
                etest = event[outer_test]
                indexes = np.flatnonzero(outer_test)
                for i, index in enumerate(indexes):
                    prediction_rows.append({"landmark": landmark, "held_out_domain": held,
                                            "model": model_name, "cell_id": data["rows"][index]["cell_id"],
                                            "event_observed": int(etest[i]),
                                            "followup_cycle": float(ttest[i] + landmark),
                                            "residual_followup": float(ttest[i]),
                                            "median_residual": float(med[i]) if np.isfinite(med[i]) else "",
                                            "median_total_cycle": float(med[i] + landmark) if np.isfinite(med[i]) else "",
                                            "risk_score": float(pred["risk_score"][i]),
                                            "aft_mu": float(pred["mu"][i]) if pred["kind"] == "lognormal" else "",
                                            "aft_sigma": pred["sigma"] if pred["kind"] == "lognormal" else "",
                                            "chosen_candidate": selected[model_name] if selected[model_name] is not None else "",
                                            **{f"risk_by_{h}": float(1 - survival_at(pred, h - landmark)[i]) if h > landmark else "" for h in HORIZONS}})
                usable = []
                for horizon in HORIZONS:
                    if horizon <= landmark:
                        continue
                    rh = horizon - landmark
                    risk = 1 - survival_at(pred, rh)
                    if not np.all((risk >= -1e-12) & (risk <= 1 + 1e-12)):
                        raise AssertionError("Survival probability out of range")
                    brier, g, n_late = legacy.ipcw_brier(ttest, etest, risk, rh)
                    # A step model does not extrapolate beyond its source observation support.
                    if pred["kind"] == "step" and rh >= pred["source_max_followup"]:
                        brier = None
                    horizon_rows.append({"landmark": landmark, "held_out_domain": held,
                                         "model": model_name, "horizon_cycle": horizon,
                                         "n_target": int(outer_test.sum()), "n_target_events": int(etest.sum()),
                                         "n_at_risk_after_horizon": n_late,
                                         "censor_survival": float(g),
                                         "supported": int(brier is not None),
                                         "ipcw_brier": brier,
                                         "mean_predicted_risk": float(risk.mean()),
                                         "km_observed_risk": float(1 - legacy.km_survival(ttest, etest, rh)) if brier is not None else "",
                                         "source_extrapolation": int(rh >= pred["source_max_followup"])})
                    if brier is not None:
                        usable.append((horizon, brier))
                concordance = concordance_index_censored(etest, ttest, np.asarray(pred["risk_score"]))[0]
                event_median = med[etest]
                observed_time = ttest[etest] + landmark
                medape = float(np.median(np.abs(event_median[np.isfinite(event_median)] + landmark - observed_time[np.isfinite(event_median)]) /
                                         observed_time[np.isfinite(event_median)] * 100)) if np.isfinite(event_median).any() else None
                ibs = None
                if len(usable) >= 2:
                    xs, ys = (np.asarray(z, dtype=float) for z in zip(*usable))
                    ibs = float(np.trapezoid(ys, xs) / (xs[-1] - xs[0]))
                fold_rows.append({"landmark": landmark, "held_out_domain": held, "model": model_name,
                                  "n_target": int(outer_test.sum()), "n_events": int(etest.sum()),
                                  "selected_candidate": selected[model_name] if selected[model_name] is not None else "",
                                  "harrell_c": float(concordance), "event_medape_pct": medape,
                                  "n_event_medians_available": int(np.isfinite(event_median).sum()),
                                  "n_supported_horizons": len(usable),
                                  "first_supported_horizon": usable[0][0] if len(usable) >= 2 else "",
                                  "last_supported_horizon": usable[-1][0] if len(usable) >= 2 else "",
                                  "grid_integrated_brier": ibs})
            print(f"L={landmark} held={held}: selected {selected}", flush=True)
    for name, rows in (("source_validation_results.csv", selection_rows),
                       ("benchmark_predictions.csv", prediction_rows),
                       ("benchmark_horizons.csv", horizon_rows),
                       ("benchmark_metrics.csv", fold_rows)):
        with (HERE / name).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (HERE / "benchmark_config.json").write_text(json.dumps({
        "protocol": "protocol_v4.json", "model_candidates": CANDIDATES,
        "xgb_boost_rounds": N_BOOST, "rsf_trees": N_TREES,
        "source_selection_score": "NLL for parametric AFT; supported Brier for Cox/RSF",
        "target_outcomes_used_for_selection": False,
        "note": "model-specific unsupported horizons are NA; cross-model paired comparisons require common supported grid",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
