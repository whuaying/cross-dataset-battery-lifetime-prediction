"""Select target-shift regularization using source domains only.

For each outer held-out source, an inner model trains on four other sources and
predicts the fifth. Random calibration/evaluation partitions of that fifth
source select one prior strength. The outer target source is never accessed.
"""
from __future__ import annotations

import csv
import importlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

base = importlib.import_module("01_residual_survival_benchmark")
recal = importlib.import_module("02_target_recalibration")
HERE = Path(__file__).resolve().parent
PRIORS = (0.25, 0.5, 1.0)
N_DRAWS = 10


def main() -> None:
    protocol = json.loads((HERE / "protocol_v4.json").read_text(encoding="utf-8"))
    if tuple(protocol["shift_prior_candidates"]) != PRIORS:
        raise AssertionError("Shift-prior candidate set differs from protocol")
    selected_scale = {}
    with (HERE / "benchmark_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        for r in csv.DictReader(stream):
            if r["model"] == "xgb_aft":
                selected_scale[(int(r["landmark"]), r["held_out_domain"])] = float(r["selected_candidate"])
    rows = []
    selected = {}
    for landmark in (50, 100):
        data = base.load(landmark)
        domains, x, event, time = (data[k] for k in ("domain", "x", "event", "time"))
        selected[str(landmark)] = {}
        for held in base.DOMAINS:
            scale = selected_scale[(landmark, held)]
            scores = defaultdict(list)
            for valid in (d for d in base.DOMAINS if d != held):
                train = (domains != held) & (domains != valid)
                evaluation = domains == valid
                pred = base.fit_predict("xgb_aft", scale, x[train], x[evaluation],
                                        base.structured(event[train], time[train]))
                mu = np.asarray(pred["mu"], dtype=float)
                ev, tm = event[evaluation], time[evaluation]
                n = len(tm)
                m = math.ceil(.2 * n)
                for draw in range(N_DRAWS):
                    rng = np.random.default_rng(20261003 + landmark * 10000 +
                                                base.DOMAINS.index(held) * 1000 +
                                                base.DOMAINS.index(valid) * 100 + draw)
                    order = rng.permutation(n)
                    cal, test = order[:m], order[m:]
                    for prior_sd in PRIORS:
                        shift = recal.fit_shift_with_prior(tm[cal], ev[cal], mu[cal], scale, prior_sd)
                        nll = float(recal.nll(tm[test], ev[test], mu[test], scale, shift).mean())
                        scores[prior_sd].append(nll)
                        rows.append({"landmark": landmark, "outer_held_out_domain": held,
                                     "source_validation_domain": valid, "draw": draw,
                                     "prior_log_shift_sd": prior_sd,
                                     "n_source_calibration": len(cal),
                                     "n_source_evaluation": len(test),
                                     "evaluation_mean_censored_nll": nll})
            if any(len(scores[p]) != 5 * N_DRAWS for p in PRIORS):
                raise AssertionError("Incomplete source-only prior selection")
            chosen = min(PRIORS, key=lambda p: (float(np.mean(scores[p])), p))
            selected[str(landmark)][held] = chosen
            print(f"L={landmark} held={held}: prior={chosen}, source NLL=" +
                  ", ".join(f"{p}:{np.mean(scores[p]):.4f}" for p in PRIORS), flush=True)
    with (HERE / "source_prior_validation.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "source_prior_selection.json").write_text(json.dumps({
        "candidates": PRIORS, "source_draws_per_validation_domain": N_DRAWS,
        "budget_fraction": .2, "selection_metric": "mean right-censored NLL across five source validation domains and fixed random draws",
        "outer_target_outcomes_used": False,
        "selected_prior_log_sd": selected,
        "note": "source-only hyperparameter selection, not a prospective deployment validation",
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
