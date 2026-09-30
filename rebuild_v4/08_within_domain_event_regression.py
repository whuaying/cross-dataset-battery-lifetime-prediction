"""Paired exploratory controls for source transfer and event-only regression.

The source-held-out AFT predictions are the frozen v4 benchmark.  The in-domain
arm predicts every target cell out of fold, using target labels only in its
other folds.  The event-only arm uses the same source cells, features, XGBoost
tree budget and target test cells as source AFT, but fits log residual lifetime
only among observed source events.  Its survival curve borrows the source-only
AFT scale; this is a specified risk proxy, not a separately calibrated model.
"""
from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored

base = importlib.import_module("01_residual_survival_benchmark")
HERE = Path(__file__).resolve().parent
ARMS = ("source_aft", "within_domain_aft", "source_event_only_regression")


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def event_only_predict(x_train: np.ndarray, x_test: np.ndarray,
                       time: np.ndarray, event: np.ndarray,
                       sigma: float, seed: int = 0) -> dict:
    if event.sum() < 2:
        raise ValueError("Event-only training requires at least two observed events")
    imputer = SimpleImputer(strategy="median").fit(x_train)
    xtr = imputer.transform(x_train)[event]
    xte = imputer.transform(x_test)
    train = xgb.DMatrix(xtr, label=np.log(time[event]), feature_names=list(base.FEATURES))
    params = {"objective": "reg:squarederror", "tree_method": "hist",
              "max_depth": 4, "eta": 0.05, "subsample": 0.8,
              "colsample_bytree": 0.8, "lambda": 1.0, "seed": seed, "nthread": 4}
    model = xgb.train(params, train, num_boost_round=base.N_BOOST, verbose_eval=False)
    mu = model.predict(xgb.DMatrix(xte, feature_names=list(base.FEATURES)))
    return {"kind": "lognormal", "mu": mu, "sigma": sigma,
            "risk_score": -mu, "source_max_followup": float(time.max()), "fitted": True}


def main() -> None:
    protocol = json.loads((HERE / "protocol_v4.json").read_text(encoding="utf-8"))
    selected = {(int(r["landmark"]), r["held_out_domain"]): float(r["selected_candidate"])
                for r in read_csv(HERE / "benchmark_metrics.csv") if r["model"] == "xgb_aft"}
    frozen = {(int(r["landmark"]), r["held_out_domain"], r["cell_id"]): r
              for r in read_csv(HERE / "benchmark_predictions.csv") if r["model"] == "xgb_aft"}
    predictions, horizons, metrics, memberships = [], [], [], []
    for landmark in protocol["landmarks"]:
        data = base.load(landmark)
        for domain_name in protocol["domains"]:
            target_idx = np.flatnonzero(data["domain"] == domain_name)
            source_idx = np.flatnonzero(data["domain"] != domain_name)
            n = len(target_idx)
            event_target = data["event"][target_idx]
            time_target = data["time"][target_idx]
            sigma = selected[(landmark, domain_name)]
            if n < 20:
                n_splits = 3
            else:
                n_splits = 5
            counts = np.bincount(event_target.astype(int), minlength=2)
            present = counts[counts > 0]
            n_splits = min(n_splits, int(present.min()))
            if n_splits < 2:
                raise RuntimeError(f"Insufficient cells for in-domain folds: {landmark}/{domain_name}")
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                       random_state=20260930 + landmark)
            arm_predictions: dict[str, dict[int, dict]] = {arm: {} for arm in ARMS}
            for fold, (local_train, local_test) in enumerate(splitter.split(target_idx, event_target)):
                train_idx, test_idx = target_idx[local_train], target_idx[local_test]
                y_train = base.structured(data["event"][train_idx], data["time"][train_idx])
                pred = base.fit_predict("xgb_aft", sigma, data["x"][train_idx],
                                        data["x"][test_idx], y_train)
                if not pred["fitted"]:
                    raise RuntimeError(f"In-domain fit failed: {landmark}/{domain_name}/{fold}")
                for j, idx in enumerate(test_idx):
                    arm_predictions["within_domain_aft"][int(idx)] = {
                        "mu": float(pred["mu"][j]), "sigma": float(pred["sigma"]),
                        "fold": fold, "n_training": len(train_idx),
                        "n_training_events": int(data["event"][train_idx].sum())}
                    memberships.append({"landmark": landmark, "domain": domain_name,
                                        "cell_id": data["rows"][idx]["cell_id"],
                                        "fold": fold, "role": "evaluation",
                                        "n_train": len(train_idx),
                                        "n_train_events": int(data["event"][train_idx].sum())})
            event_pred = event_only_predict(data["x"][source_idx], data["x"][target_idx],
                                            data["time"][source_idx], data["event"][source_idx], sigma)
            for j, idx in enumerate(target_idx):
                key = (landmark, domain_name, data["rows"][idx]["cell_id"])
                old = frozen[key]
                if int(old["event_observed"]) != int(data["event"][idx]) or not np.isclose(
                        float(old["residual_followup"]), data["time"][idx]):
                    raise AssertionError("Frozen benchmark and current cohort differ")
                arm_predictions["source_aft"][int(idx)] = {
                    "mu": float(old["aft_mu"]), "sigma": float(old["aft_sigma"]),
                    "fold": "", "n_training": len(source_idx),
                    "n_training_events": int(data["event"][source_idx].sum())}
                arm_predictions["source_event_only_regression"][int(idx)] = {
                    "mu": float(event_pred["mu"][j]), "sigma": sigma,
                    "fold": "", "n_training": int(data["event"][source_idx].sum()),
                    "n_training_events": int(data["event"][source_idx].sum())}
                frozen_pred = {"kind": "lognormal", "mu": np.array([float(old["aft_mu"])]),
                               "sigma": float(old["aft_sigma"]), "risk_score": np.zeros(1)}
                for h in base.HORIZONS:
                    if h > landmark and not np.isclose(1-base.survival_at(frozen_pred, h-landmark)[0],
                                                      float(old[f"risk_by_{h}"]), atol=1e-10):
                        raise AssertionError("Frozen source AFT risk does not reproduce")
            if any(set(values) != set(map(int, target_idx)) for values in arm_predictions.values()):
                raise AssertionError("An arm is missing a target cell")
            for arm in ARMS:
                values = [arm_predictions[arm][int(idx)] for idx in target_idx]
                mu = np.array([v["mu"] for v in values])
                scale = np.array([v["sigma"] for v in values])
                if not np.allclose(scale, sigma):
                    raise AssertionError("Source-only AFT scale differs across paired arms")
                risk_score = -mu
                med = np.exp(mu)
                pred = {"kind": "lognormal", "mu": mu, "sigma": sigma,
                        "risk_score": risk_score}
                for j, idx in enumerate(target_idx):
                    row = {"landmark": landmark, "domain": domain_name, "cell_id": data["rows"][idx]["cell_id"],
                           "arm": arm, "event_observed": int(data["event"][idx]),
                           "residual_followup": float(data["time"][idx]),
                           "fold": values[j]["fold"], "n_training": values[j]["n_training"],
                           "n_training_events": values[j]["n_training_events"],
                           "log_median_residual": float(mu[j]), "sigma": sigma,
                           "median_total_cycle": float(landmark + med[j])}
                    row.update({f"risk_by_{h}": float(1 - base.survival_at(pred, h-landmark)[j])
                                if h > landmark else "" for h in base.HORIZONS})
                    predictions.append(row)
                supported = []
                for h in base.HORIZONS:
                    if h <= landmark:
                        continue
                    risk = 1 - base.survival_at(pred, h-landmark)
                    score, g, n_late = base.legacy.ipcw_brier(time_target, event_target, risk, h-landmark)
                    horizons.append({"landmark": landmark, "domain": domain_name, "arm": arm,
                                     "horizon_cycle": h, "n_at_risk_after_horizon": n_late,
                                     "censor_survival": g, "ipcw_brier": score,
                                     "mean_predicted_risk": float(risk.mean())})
                    if score is not None:
                        supported.append((h, score))
                if len(supported) >= 2:
                    grid, scores = (np.array(z, dtype=float) for z in zip(*supported))
                    ibs = float(np.trapezoid(scores, grid) / (grid[-1]-grid[0]))
                else:
                    ibs = None
                event_median = med[event_target]
                event_total = time_target[event_target] + landmark
                medape = float(np.median(np.abs(event_median + landmark - event_total) / event_total) * 100)
                c_index = float(concordance_index_censored(event_target, time_target, risk_score)[0])
                metrics.append({"landmark": landmark, "domain": domain_name, "arm": arm,
                                "n_target": n, "n_events": int(event_target.sum()),
                                "n_source_train": len(source_idx),
                                "n_source_events": int(data["event"][source_idx].sum()),
                                "n_in_domain_folds": n_splits if arm == "within_domain_aft" else "",
                                "n_supported_horizons": len(supported),
                                "first_supported_horizon": supported[0][0] if len(supported) >= 2 else "",
                                "last_supported_horizon": supported[-1][0] if len(supported) >= 2 else "",
                                "grid_integrated_brier": ibs,
                                "harrell_c": c_index, "event_medape_pct": medape})
            print(f"L={landmark} {domain_name}: n={n}, events={event_target.sum()}, folds={n_splits}", flush=True)
    for name, records in (("evidence08_predictions.csv", predictions),
                          ("evidence08_horizons.csv", horizons),
                          ("evidence08_metrics.csv", metrics),
                          ("evidence08_membership.csv", memberships)):
        write_csv(HERE / name, records)
    input_names = ["protocol_v4.json", "benchmark_metrics.csv", "benchmark_predictions.csv"]
    input_hashes = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in input_names}
    (HERE / "evidence08_protocol.json").write_text(json.dumps({
        "status": "post-v4 exploratory evidence; not preregistered",
        "landmarks": protocol["landmarks"], "domains": protocol["domains"],
        "feature_set": list(base.FEATURES), "endpoint": protocol["endpoint"],
        "source_aft": "frozen v4 source-held-out XGBoost AFT cell predictions",
        "within_domain_aft": "stratified 3- or 5-fold target-cell out-of-fold XGBoost AFT",
        "within_domain_hyperparameter": "same scale selected without target labels for source AFT",
        "event_only": "source-domain observed events only, XGBoost reg:squarederror on log residual lifetime",
        "event_only_risk_distribution": "lognormal proxy using the paired source AFT sigma, not separately calibrated",
        "xgboost_budget": {"rounds": base.N_BOOST, "max_depth": 4, "eta": 0.05,
                            "subsample": 0.8, "colsample_bytree": 0.8, "lambda": 1.0},
        "in_domain_label_access": "other target folds supply event and censor labels; zero-shot source AFT has no target labels",
        "score": "same held-out cells and target censoring weights per domain; domain-specific supported absolute horizons",
        "small_source_warning": "NASA and CALCE out-of-fold fits use 5-7 in-domain training cells and are descriptive only",
        "input_sha256": input_hashes,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
