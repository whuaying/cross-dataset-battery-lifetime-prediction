"""Within-source held-out protocol-group AFT control for XJTU and TJU.

The available XJTU batch and TJU dataset-subset fields define groups.  Every
target cell receives one prediction from a model trained on the other groups
of the same source.  The source-held-out model's frozen predictions provide a
paired reference on exactly those cells.  TJU groups also differ in chemistry,
so this is a compound group shift rather than an isolated protocol effect.
"""
from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path

import numpy as np

base = importlib.import_module("01_residual_survival_benchmark")
evidence = importlib.import_module("08_within_domain_event_regression")
HERE = Path(__file__).resolve().parent
DOMAINS = ("XJTU", "TJU")


def main() -> None:
    selected = {(int(r["landmark"]), r["held_out_domain"]): float(r["selected_candidate"])
                for r in evidence.read_csv(HERE / "benchmark_metrics.csv") if r["model"] == "xgb_aft"}
    frozen = {(int(r["landmark"]), r["held_out_domain"], r["cell_id"]): r
              for r in evidence.read_csv(HERE / "benchmark_predictions.csv") if r["model"] == "xgb_aft"}
    group_rows, pred_rows, score_rows, horizon_rows = [], [], [], []
    for landmark in (50, 100):
        data = base.load(landmark)
        for domain_name in DOMAINS:
            target = np.flatnonzero(data["domain"] == domain_name)
            groups = np.array([data["rows"][i]["protocol"] for i in target])
            if len(set(groups)) < 3:
                raise AssertionError("Insufficient protocol groups")
            sigma = selected[(landmark, domain_name)]
            mu = np.full(len(target), np.nan)
            source_mu = np.full(len(target), np.nan)
            for group in sorted(set(groups)):
                local_train = groups != group
                local_test = groups == group
                train_idx, test_idx = target[local_train], target[local_test]
                group_rows.append({"landmark": landmark, "domain": domain_name,
                                   "held_out_group": group, "n_train": len(train_idx),
                                   "n_train_events": int(data["event"][train_idx].sum()),
                                   "n_test": len(test_idx),
                                   "n_test_events": int(data["event"][test_idx].sum()),
                                   "train_chemistries": ";".join(sorted({data["rows"][i]["chemistry"] for i in train_idx})),
                                   "test_chemistries": ";".join(sorted({data["rows"][i]["chemistry"] for i in test_idx}))})
                fitted = base.fit_predict("xgb_aft", sigma, data["x"][train_idx],
                                          data["x"][test_idx],
                                          base.structured(data["event"][train_idx], data["time"][train_idx]))
                if not fitted["fitted"]:
                    raise RuntimeError(f"Protocol-held-out fit failed: {landmark}/{domain_name}/{group}")
                mu[local_test] = fitted["mu"]
                for local_i, idx in enumerate(test_idx):
                    key = (landmark, domain_name, data["rows"][idx]["cell_id"])
                    ref = frozen[key]
                    source_mu[np.flatnonzero(target == idx)[0]] = float(ref["aft_mu"])
            if not np.isfinite(mu).all() or not np.isfinite(source_mu).all():
                raise AssertionError("Incomplete protocol predictions")
            for arm, location in (("within_source_held_protocol", mu), ("source_held_out", source_mu)):
                pred = {"kind": "lognormal", "mu": location, "sigma": sigma,
                        "risk_score": -location}
                usable = []
                for h in base.HORIZONS:
                    if h <= landmark:
                        continue
                    risk = 1-base.survival_at(pred, h-landmark)
                    brier, g, n_late = base.legacy.ipcw_brier(data["time"][target],
                                                                 data["event"][target], risk, h-landmark)
                    horizon_rows.append({"landmark": landmark, "domain": domain_name,
                                         "arm": arm, "horizon_cycle": h, "ipcw_brier": brier,
                                         "censor_survival": g, "n_at_risk_after_horizon": n_late})
                    if brier is not None:
                        usable.append((h, brier))
                if len(usable) >= 2:
                    grid, score = (np.asarray(x, dtype=float) for x in zip(*usable))
                    ibs = float(np.trapezoid(score, grid)/(grid[-1]-grid[0]))
                else:
                    ibs = None
                score_rows.append({"landmark": landmark, "domain": domain_name,
                                   "arm": arm, "n_target": len(target),
                                   "n_events": int(data["event"][target].sum()),
                                   "n_groups": len(set(groups)), "n_supported_horizons": len(usable),
                                   "first_supported_horizon": usable[0][0] if len(usable) >= 2 else "",
                                   "last_supported_horizon": usable[-1][0] if len(usable) >= 2 else "",
                                   "grid_integrated_brier": ibs})
                for j, idx in enumerate(target):
                    pred_rows.append({"landmark": landmark, "domain": domain_name,
                                      "cell_id": data["rows"][idx]["cell_id"],
                                      "held_out_group": groups[j], "arm": arm,
                                      "event_observed": int(data["event"][idx]),
                                      "residual_followup": float(data["time"][idx]),
                                      "log_median_residual": float(location[j]),
                                      "median_total_cycle": float(landmark+np.exp(location[j]))})
            print(f"L={landmark} {domain_name}: {len(target)} target cells, {len(set(groups))} groups", flush=True)
    for name, records in (("evidence08p_groups.csv", group_rows),
                          ("evidence08p_predictions.csv", pred_rows),
                          ("evidence08p_horizons.csv", horizon_rows),
                          ("evidence08p_metrics.csv", score_rows)):
        evidence.write_csv(HERE / name, records)
    (HERE / "evidence08p_protocol.json").write_text(json.dumps({
        "status": "exploratory; groups come from verified-v3 protocol field",
        "domains": DOMAINS, "landmarks": [50, 100],
        "XJTU_groups": "Batch-1 through Batch-4; operational Batch-5/6 lack landmark diagnostic density",
        "TJU_groups": "Dataset_1_NCA, Dataset_2_NCM, Dataset_3_NCM_NCA; protocol and chemistry confounded",
        "train": "all other groups of the target source; cell-level labels available only in these groups",
        "test": "one held-out group; all group predictions pooled for source-level paired scoring",
        "model": "residual-life XGBoost AFT with source-only selected scale and frozen v4 features/rounds",
        "interpretation": "not an isolated protocol causal effect; MIT/HUST lack usable group metadata and NASA is too small",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
