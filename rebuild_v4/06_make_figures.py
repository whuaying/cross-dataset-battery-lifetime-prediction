"""Publication-oriented exploratory v4 figures from machine-readable outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent.parent / "figures_v4"
OUT.mkdir(exist_ok=True)
DOMAINS = ("MIT", "XJTU", "TJU", "NASA", "CALCE", "HUST")
# Colorblind-safe Okabe-Ito-inspired colors; labels and markers also identify series.
COLORS = {"km": "#7f7f7f", "xgb_aft": "#0072B2", "lognormal_aft": "#CC79A7",
          "cox": "#D55E00", "rsf": "#009E73"}
DOMAIN_COLORS = {"MIT": "#0072B2", "XJTU": "#E69F00", "TJU": "#009E73",
                 "NASA": "#56B4E9", "CALCE": "#D55E00", "HUST": "#CC79A7"}
DOMAIN_MARKERS = {"MIT": "o", "XJTU": "s", "TJU": "^", "NASA": "D", "CALCE": "P", "HUST": "X"}


def csv_rows(name: str) -> list[dict]:
    with (HERE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def source_cohort() -> None:
    with (HERE.parent / "rebuild_v2" / "cell_audit.csv").open(encoding="utf-8-sig", newline="") as stream:
        audit = list(csv.DictReader(stream))
    counts = {}
    for domain in DOMAINS:
        group = [r for r in audit if r["dataset"] == domain]
        eligible = [r for r in group if r["landmark_100_status"] == "eligible"]
        counts[domain] = {
            "eligible": len(eligible),
            "pre_eol": sum(r["landmark_100_status"] == "eol_before_or_at_landmark" for r in group),
            "no_followup": sum(r["landmark_100_status"] == "no_followup_after_landmark" for r in group),
            "too_few": sum(r["landmark_100_status"] == "too_few_pre_landmark_observations" for r in group),
            "events": sum(r["event_observed"] == "1" for r in eligible),
        }
    y = np.arange(len(DOMAINS))
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    labels = (("events", "Event after landmark", "#009E73", ""),
              ("censored", "Right-censored", "#E69F00", "///"),
              ("pre_eol", "EOL by landmark", "#D55E00", "xx"),
              ("no_followup", "No follow-up after landmark", "#7f7f7f", "..."),
              ("too_few", "Too few early records", "#CC79A7", "\\\\"))
    left = np.zeros(len(DOMAINS))
    for key, label, color, hatch in labels:
        vals = np.array([counts[d][key] if key != "censored" else
                         counts[d]["eligible"] - counts[d]["events"] for d in DOMAINS])
        ax.barh(y, vals, left=left, label=label, color=color, height=0.68,
                hatch=hatch, edgecolor="white", linewidth=0.5)
        for i, value in enumerate(vals):
            if value >= 8:
                ax.text(left[i] + value / 2, i, str(value), ha="center", va="center",
                        fontsize=8, color="white" if color != "#E69F00" else "#263645")
        left += vals
    ax.set_yticks(y, DOMAINS)
    ax.invert_yaxis()
    ax.set_xlim(0, 142)
    ax.set_xlabel("Verified unique cells")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    for i, total in enumerate(left):
        ax.text(total + 1.5, i, str(int(total)), va="center", fontsize=8)
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.set_title("Six-source cycle-100 cohort: eligibility and outcome")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, "fig_v4_source_cohort")


def benchmark() -> None:
    records = [r for r in csv_rows("benchmark_metrics.csv") if r["landmark"] == "100"]
    models = ("km", "lognormal_aft", "cox", "rsf", "xgb_aft")
    labels = {"km": "Source KM", "lognormal_aft": "Lognormal AFT",
              "cox": "Cox", "rsf": "Survival forest", "xgb_aft": "XGBoost AFT"}
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.3))
    for ax, domain in zip(axes.flat, DOMAINS):
        fold = {r["model"]: r for r in records if r["held_out_domain"] == domain}
        ax.set_title(f"{domain} (n={fold['km']['n_target']}, events={fold['km']['n_events']})")
        if domain == "NASA":
            ax.text(0.5, 0.5, "Integrated Brier\nunavailable", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            continue
        y = np.arange(len(models))
        vals = [float(fold[m]["grid_integrated_brier"]) for m in models]
        ax.barh(y, vals, color=[COLORS[m] for m in models], alpha=0.82)
        ax.set_yticks(y, [labels[m] for m in models], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(0.38, max(vals) * 1.12))
        ax.grid(axis="x", alpha=0.2)
        for i, value in enumerate(vals):
            ax.text(value + 0.005, i, f"{value:.3f}", va="center", fontsize=7)
    fig.supxlabel("Integrated IPCW Brier over each domain's supported cycle grid (lower is better)")
    fig.suptitle("Cycle-100 source-held-out residual-life benchmark", fontsize=12)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.95))
    save(fig, "fig_v4_benchmark_brier")


def endpoint() -> None:
    records = [r for r in csv_rows("endpoint_model_metrics.csv") if r["landmark"] == "100"]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2))
    for ax, domain in zip(axes.flat, DOMAINS):
        ax.set_title(domain)
        if domain == "NASA":
            ax.text(0.5, 0.5, "Integrated Brier\nunavailable", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_xticks([1,3,5])
            continue
        for model, name in (("cox", "Cox"), ("xgb_aft", "XGBoost AFT")):
            group = sorted([r for r in records if r["domain"] == domain and r["model"] == model],
                           key=lambda r: int(r["endpoint_records"]))
            ax.plot([int(r["endpoint_records"]) for r in group],
                    [float(r["common_grid_integrated_brier"]) for r in group],
                    "s--" if model == "cox" else "o-", label=name, color=COLORS[model],
                    linewidth=1.7, markersize=4)
        ax.set_xticks([1,3,5])
        ax.grid(alpha=0.2)
        if domain == "MIT":
            ax.legend(fontsize=8)
    fig.supxlabel("Consecutive observed SOH ≤ 0.8 records defining EOL")
    fig.supylabel("Integrated Brier on common cells and shared horizons")
    fig.suptitle("Endpoint-definition sensitivity; each rule defines a different outcome", fontsize=12)
    fig.tight_layout(rect=(0.03, 0.03, 1, 0.95))
    save(fig, "fig_v4_endpoint_model_sensitivity")


def recalibration() -> None:
    records = [r for r in csv_rows("recalibration_scores.csv") if r["landmark"] == "100" and
               abs(float(r["budget_fraction"]) - 0.2) < 1e-10]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 6.2))
    for ax, domain in zip(axes.flat, DOMAINS):
        group = [r for r in records if r["domain"] == domain]
        vals = [float(r["grid_brier_difference"]) for r in group if r["grid_brier_difference"]]
        ax.axhline(0, color="0.4", linewidth=1)
        if vals:
            ax.boxplot(vals, widths=0.38, patch_artist=True,
                       boxprops={"facecolor": "#8ebbd5", "edgecolor": "#3973b9"},
                       medianprops={"color": "#b2463d"})
            ax.scatter(1 + np.linspace(-0.13, 0.13, len(vals)), vals,
                       s=9, color="#3973b9", alpha=0.35)
            ax.set_xticks([1], [f"{sum(x<0 for x in vals)}/{len(vals)} improve"])
        else:
            ax.text(0.5, 0.5, "Integrated Brier\nunavailable", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_xticks([])
        total = int(group[0]["n_calibration"]) + int(group[0]["n_evaluation"])
        ax.set_title(f"{domain} (cal {group[0]['n_calibration']}, total {total})")
        ax.grid(axis="y", alpha=0.2)
    fig.supylabel("Shifted minus baseline integrated Brier; negative favors adaptation")
    fig.suptitle("Cycle-100 random target-cell draws: 20% calibration, 30 repeats", fontsize=12)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.95))
    save(fig, "fig_v4_target_recalibration")


def source_correction() -> None:
    records = [r for r in csv_rows("source_correction_metrics.csv") if r["landmark"] == "100"]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    xpos = np.arange(len(DOMAINS))
    for j, model in enumerate(("xgb_aft", "cox")):
        vals = []
        for domain in DOMAINS:
            pair = {r["training_variant"]: r for r in records if r["held_out_domain"] == domain and r["model"] == model}
            if not pair["certified_v3"]["grid_integrated_brier"] or not pair["before_source_correction"]["grid_integrated_brier"]:
                vals.append(np.nan)
            else:
                vals.append(float(pair["certified_v3"]["grid_integrated_brier"]) -
                            float(pair["before_source_correction"]["grid_integrated_brier"]))
        ax.bar(xpos + (j-0.5)*0.29, vals, width=0.27, label="XGBoost AFT" if j==0 else "Cox",
               color=COLORS[model], hatch="" if j == 0 else "///",
               edgecolor="black", linewidth=0.35)
    ax.axhline(0, color="0.3", linewidth=1)
    ax.set_xticks(xpos, DOMAINS)
    ax.set_ylabel("Certified minus historical-training integrated Brier")
    ax.set_title("Cycle-100 training-data correction on fixed certified test labels")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    save(fig, "fig_v4_source_correction")


def survival_calibration() -> None:
    records = [r for r in csv_rows("benchmark_horizons.csv") if r["landmark"] == "100" and
               r["model"] == "xgb_aft" and r["supported"] == "1"]
    fig, ax = plt.subplots(figsize=(6.1, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="0.4", linewidth=1, label="Equality")
    for domain in DOMAINS:
        group = [r for r in records if r["held_out_domain"] == domain]
        ax.scatter([float(r["mean_predicted_risk"]) for r in group],
                   [float(r["km_observed_risk"]) for r in group],
                   color=DOMAIN_COLORS[domain], marker=DOMAIN_MARKERS[domain],
                   edgecolor="black", linewidth=0.35, s=42,
                   label=f"{domain} ({len(group)} horizon{'s' if len(group) != 1 else ''})", alpha=0.8)
        for r in group:
            target = (domain, int(r["horizon_cycle"]))
            if target in {("CALCE", 500), ("TJU", 1000), ("HUST", 1000), ("HUST", 2000)}:
                note = (f"CALCE 500\n{float(r['mean_predicted_risk']):.3f} / {float(r['km_observed_risk']):.3f}"
                        if target == ("CALCE", 500) else
                        f"HUST 2000\n{float(r['mean_predicted_risk']):.3f} / {float(r['km_observed_risk']):.3f}"
                        if target == ("HUST", 2000) else r["horizon_cycle"])
                offset = (-98, 16) if target == ("CALCE", 500) else (8, -25) if target == ("HUST", 2000) else (5, 3)
                ax.annotate(note,
                            (float(r["mean_predicted_risk"]), float(r["km_observed_risk"])),
                            xytext=offset, textcoords="offset points", fontsize=7,
                            arrowprops={"arrowstyle": "-", "color": "0.4", "lw": 0.6} if target in {("CALCE", 500), ("HUST", 2000)} else None)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean XGBoost AFT predicted event risk",
           ylabel="Held-out Kaplan--Meier event risk",
           title="Cycle-100 residual-life risk audit")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_v4_survival_calibration")


def main() -> None:
    source_cohort()
    benchmark()
    endpoint()
    recalibration()
    source_correction()
    survival_calibration()
    print(f"Saved six paired PDF/PNG figures in {OUT}")


if __name__ == "__main__":
    main()
