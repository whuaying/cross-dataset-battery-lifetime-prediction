"""Descriptive partition/seed sensitivity for the cycle-100 evidence arms."""
from __future__ import annotations

import csv
import importlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

base = importlib.import_module("01_residual_survival_benchmark")
evidence = importlib.import_module("08_within_domain_event_regression")
HERE = Path(__file__).resolve().parent
LANDMARK = 100


def ibs(pred: dict, time: np.ndarray, event: np.ndarray) -> tuple[float | None, int]:
    values = []
    for h in base.HORIZONS:
        if h <= LANDMARK:
            continue
        risk = 1-base.survival_at(pred, h-LANDMARK)
        score, _, _ = base.legacy.ipcw_brier(time, event, risk, h-LANDMARK)
        if score is not None:
            values.append((h, score))
    if len(values) < 2:
        return None, len(values)
    grid, score = (np.array(a, dtype=float) for a in zip(*values))
    return float(np.trapezoid(score, grid)/(grid[-1]-grid[0])), len(values)


def main() -> None:
    data = base.load(LANDMARK)
    chosen = {(int(r["landmark"]), r["held_out_domain"]): float(r["selected_candidate"])
              for r in evidence.read_csv(HERE / "benchmark_metrics.csv") if r["model"] == "xgb_aft"}
    aft_seed_scores = {(r["held_out_domain"], int(r["seed"])): float(r["grid_integrated_brier"])
                       for r in evidence.read_csv(HERE / "seed_metrics.csv")
                       if int(r["landmark"]) == LANDMARK and r["grid_integrated_brier"]}
    base_scores = {r["domain"]: float(r["grid_integrated_brier"])
                   for r in evidence.read_csv(HERE / "evidence08_metrics.csv")
                   if int(r["landmark"]) == LANDMARK and r["arm"] == "source_aft"
                   and r["grid_integrated_brier"]}
    records = []
    for domain in base.DOMAINS:
        target = np.flatnonzero(data["domain"] == domain)
        source = np.flatnonzero(data["domain"] != domain)
        time, event = data["time"][target], data["event"][target]
        sigma = chosen[(LANDMARK, domain)]
        n_splits = 3 if len(target) < 20 else 5
        class_counts = np.bincount(event.astype(int), minlength=2)
        n_splits = min(n_splits, int(class_counts[class_counts > 0].min()))
        for repetition in range(10):
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                       random_state=20260930 + LANDMARK + repetition)
            mu = np.full(len(target), np.nan)
            for train_local, test_local in splitter.split(target, event):
                train_idx, test_idx = target[train_local], target[test_local]
                fitted = base.fit_predict("xgb_aft", sigma, data["x"][train_idx],
                                          data["x"][test_idx],
                                          base.structured(data["event"][train_idx], data["time"][train_idx]),
                                          seed=0)
                mu[test_local] = fitted["mu"]
            if not np.isfinite(mu).all():
                raise AssertionError("Incomplete out-of-fold prediction")
            fitted = {"kind": "lognormal", "mu": mu, "sigma": sigma,
                      "risk_score": -mu}
            score, n_grid = ibs(fitted, time, event)
            records.append({"domain": domain, "arm": "within_domain_aft", "repetition": repetition,
                            "grid_integrated_brier": score,
                            "source_aft_paired_brier": base_scores.get(domain, ""),
                            "difference_from_source_aft": score-base_scores[domain]
                            if score is not None else "", "n_supported_horizons": n_grid})
        for seed in range(20):
            fitted = evidence.event_only_predict(data["x"][source], data["x"][target],
                                                  data["time"][source], data["event"][source],
                                                  sigma, seed=seed)
            score, n_grid = ibs(fitted, time, event)
            baseline = aft_seed_scores.get((domain, seed))
            records.append({"domain": domain, "arm": "source_event_only_regression", "repetition": seed,
                            "grid_integrated_brier": score,
                            "source_aft_paired_brier": baseline if baseline is not None else "",
                            "difference_from_source_aft": score-baseline
                            if score is not None and baseline is not None else "",
                            "n_supported_horizons": n_grid})
        print(f"cycle 100 {domain}: 10 partition fits and 20 paired model seeds", flush=True)
    evidence.write_csv(HERE / "evidence08_sensitivity.csv", records)
    summary = {}
    for (domain, arm) in {(r["domain"], r["arm"]) for r in records}:
        subset = [r for r in records if r["domain"] == domain and r["arm"] == arm]
        scores = [r["grid_integrated_brier"] for r in subset if r["grid_integrated_brier"] is not None]
        diffs = [r["difference_from_source_aft"] for r in subset
                 if r["difference_from_source_aft"] != ""]
        summary.setdefault(domain, {})[arm] = {
            "n_repetitions": len(subset), "n_integrals": len(scores),
            "score_min_max": [min(scores), max(scores)] if scores else None,
            "difference_min_max": [min(diffs), max(diffs)] if diffs else None,
            "n_lower_brier_than_paired_source_aft": sum(d < 0 for d in diffs),
            "interpretation": "overlapping cell partitions or training seeds; not independent replicates or a confidence interval",
        }
    (HERE / "evidence08_sensitivity_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
