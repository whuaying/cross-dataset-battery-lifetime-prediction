"""Check central manuscript numbers and hash the reproducible v4 artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def rows(name: str) -> list[dict]:
    with (HERE / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def close(actual: float, expected: float, tolerance: float = 5e-4) -> None:
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise AssertionError(f"Manuscript number differs: {actual} versus {expected}")


def main() -> None:
    benchmark = {(r["held_out_domain"], r["model"]): r for r in rows("benchmark_metrics.csv")
                 if r["landmark"] == "100"}
    expected_brier = {
        ("MIT", "rsf"): .139, ("TJU", "rsf"): .155,
        ("XJTU", "xgb_aft"): .055, ("HUST", "xgb_aft"): .117,
        ("CALCE", "km"): .040, ("CALCE", "xgb_aft"): .254,
    }
    for key, value in expected_brier.items():
        close(float(benchmark[key]["grid_integrated_brier"]), value)
    for model in {"km", "lognormal_aft", "cox", "rsf", "xgb_aft"}:
        assert benchmark[("NASA", model)]["grid_integrated_brier"] == ""
        assert int(benchmark[("NASA", model)]["n_supported_horizons"]) == 1
    assert sum(int(benchmark[(d, "km")]["n_target"]) for d in
               ("MIT", "XJTU", "TJU", "NASA", "CALCE", "HUST")) == 376
    assert sum(int(benchmark[(d, "km")]["n_events"]) for d in
               ("MIT", "XJTU", "TJU", "NASA", "CALCE", "HUST")) == 228
    for domain, (c_index, medape) in {
        "MIT": (.685, 198.3), "XJTU": (.491, 190.5), "TJU": (.538, 49.2),
        "NASA": (.778, 201.4), "CALCE": (.778, 49.2), "HUST": (.656, 50.3)
    }.items():
        close(float(benchmark[(domain, "xgb_aft")]["harrell_c"]), c_index)
        close(float(benchmark[(domain, "xgb_aft")]["event_medape_pct"]), medape, .05)

    with (HERE.parent / "rebuild_v2" / "cell_audit.csv").open(encoding="utf-8-sig", newline="") as stream:
        nasa = [r for r in csv.DictReader(stream) if r["dataset"] == "NASA"]
    assert len(nasa) == 34
    assert sum(r["landmark_100_status"] == "eligible" for r in nasa) == 8
    assert sum(r["landmark_100_status"] == "eol_before_or_at_landmark" for r in nasa) == 11
    assert sum(r["landmark_100_status"] == "no_followup_after_landmark" for r in nasa) == 15

    horizons = rows("endpoint_model_horizons.csv")
    endpoint = rows("endpoint_model_metrics.csv")
    xjtu_cox = {r["endpoint_records"]: r for r in endpoint if
                r["landmark"] == "100" and r["domain"] == "XJTU" and r["model"] == "cox"}
    close(float(xjtu_cox["3"]["common_grid_integrated_brier"]), .082, .0005)
    close(float(xjtu_cox["5"]["common_grid_integrated_brier"]), .071, .0005)
    for domain in ("MIT", "XJTU", "TJU", "NASA", "CALCE", "HUST"):
        for model in ("xgb_aft", "cox"):
            group = [r for r in endpoint if r["landmark"] == "100" and
                     r["domain"] == domain and r["model"] == model]
            assert {r["endpoint_records"] for r in group} == {"1", "3", "5"}
            shared = set.intersection(*(set(int(h["horizon_cycle"]) for h in horizons
                                           if h["landmark"] == "100" and h["domain"] == domain and
                                           h["model"] == model and h["endpoint_records"] == rule and
                                           h["ipcw_brier"] != "") for rule in ("1", "3", "5")))
            assert all(int(r["n_common_supported_horizons"]) == len(shared) for r in group)
            assert all((r["common_grid_integrated_brier"] != "") == (len(shared) >= 2) for r in group)

    scores = [r for r in rows("recalibration_scores.csv") if r["landmark"] == "100" and
              r["budget_fraction"] == "0.2"]
    expected_improved = {"MIT": 5, "XJTU": 6, "TJU": 13, "NASA": None,
                         "CALCE": 29, "HUST": 0}
    for domain, count in expected_improved.items():
        group = [r for r in scores if r["domain"] == domain and r["grid_brier_difference"] != ""]
        assert len(group) == (0 if count is None else 30)
        if count is not None:
            assert sum(float(r["grid_brier_difference"]) < 0 for r in group) == count

    seed_summary = json.loads((HERE / "seed_target_recalibration_summary.json").read_text(encoding="utf-8"))["100"]
    for domain, value in {"MIT": .0031, "XJTU": .0076, "TJU": .0040,
                          "CALCE": -.0757, "HUST": .0079}.items():
        close(seed_summary[domain]["mean_paired_brier_difference"], value, 5e-5)
        assert seed_summary[domain]["n_supported_paired_brier"] == 600
    assert seed_summary["NASA"]["n_supported_paired_brier"] == 0

    controls = {(r["domain"], r["arm"]): r for r in rows("evidence08_metrics.csv")
                if r["landmark"] == "100"}
    expected_controls = {
        ("MIT", "within_domain_aft"): .060, ("MIT", "source_event_only_regression"): .141,
        ("XJTU", "within_domain_aft"): .087, ("XJTU", "source_event_only_regression"): .062,
        ("TJU", "within_domain_aft"): .165, ("TJU", "source_event_only_regression"): .144,
        ("CALCE", "within_domain_aft"): .092, ("CALCE", "source_event_only_regression"): .282,
        ("HUST", "within_domain_aft"): .121, ("HUST", "source_event_only_regression"): .436,
    }
    for key, value in expected_controls.items():
        close(float(controls[key]["grid_integrated_brier"]), value)
    for domain in ("MIT", "XJTU", "TJU", "NASA", "CALCE", "HUST"):
        actual = controls[(domain, "source_aft")]["grid_integrated_brier"]
        previous = benchmark[(domain, "xgb_aft")]["grid_integrated_brier"]
        assert actual == previous
    assert all(controls[("NASA", arm)]["grid_integrated_brier"] == ""
               for arm in ("source_aft", "within_domain_aft", "source_event_only_regression"))
    group_scores = {(r["landmark"], r["domain"], r["arm"]): r
                    for r in rows("evidence08p_metrics.csv")}
    for key, value in {
        ("100", "XJTU", "within_source_held_protocol"): .058,
        ("100", "TJU", "within_source_held_protocol"): .247,
        ("50", "XJTU", "within_source_held_protocol"): .079,
        ("50", "TJU", "within_source_held_protocol"): .289,
    }.items():
        close(float(group_scores[key]["grid_integrated_brier"]), value)
    sensitivity = json.loads((HERE / "evidence08_sensitivity_summary.json").read_text(encoding="utf-8"))
    assert sensitivity["TJU"]["source_event_only_regression"]["n_lower_brier_than_paired_source_aft"] == 20
    assert sensitivity["HUST"]["source_event_only_regression"]["n_lower_brier_than_paired_source_aft"] == 0

    audit = []
    for path in sorted(HERE.iterdir()):
        if (path.is_file() and path.suffix in {".py", ".csv", ".json", ".txt", ".md"}
                and path.name != "v4_artifact_manifest.csv"):
            audit.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                          "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for path in sorted((ROOT / "figures_v4").glob("*")):
        if path.suffix in {".pdf", ".png"}:
            audit.append({"path": str(path.relative_to(ROOT)).replace("\\", "/"),
                          "bytes": path.stat().st_size,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for name in ("AE_paper_elsarticle_v4.tex", "AE_paper_elsarticle_v4.pdf",
                 "AE_V4_REVISION_REPORT.md", "AE_REVIEW_REVISION_EXECUTION_PLAN.md",
                 "AE_PRE_SUBMISSION_EVIDENCE_IMPLEMENTATION.md", "AE_SUBMISSION_V4_README.md",
                 "AE论文_CoverLetter_v4.md", "AE论文_CoverLetter.md",
                 "AE论文_Highlights_v4.txt", "AE论文_Highlights_v4.docx",
                 "AE论文_Supplementary_v4.md", "AE论文_Supplementary_v4.pdf"):
        path = ROOT / name
        audit.append({"path": name, "bytes": path.stat().st_size,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    with (HERE / "v4_artifact_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(audit)
    print(f"PASS: manuscript claims, shared endpoint horizons, and {len(audit)} artifact hashes")


if __name__ == "__main__":
    main()
