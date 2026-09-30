# Supplementary material — v4 review draft

This document accompanies `AE_paper_elsarticle_v4.tex` and the machine-readable outputs in `eda/rebuild_v4`. All analyses are retrospective and exploratory. The source-level integrated IPCW Brier score uses each source's own supported horizon grid; it is not a common cross-source integral.

## S1. Source characteristics and landmark eligibility

The underlying row-level audit is `eda/rebuild_v2/cell_audit.csv`; the eligible cell features are in `eda/rebuild_v2/landmark_100_features.csv`. The observation-density column below is the median ratio of verified unique cycle records to the integer-cycle span from first to last record across all unique cells in a source. It describes the processed record grid, not the original instrument sampling frequency. Empty temperature entries mean the verified local table did not certify a value.

| Source | Verified cells | Cycle-100 eligible/events | Recorded chemistry | Recorded temperature (°C) | Available group or protocol labels | Median record density |
|---|---:|---:|---|---|---|---:|
| MIT | 124 | 124/43 | LFP | 30 | one selected group (`b`) | 1.000 |
| XJTU | 55 | 39/21 | NCM532 | unavailable | six batch labels | 1.000 |
| TJU | 130 | 119/75 | NCA, NCM, mixed NCM/NCA | 25, 35, 45 | three data subsets; chemistry and protocol confounded | 1.000 |
| NASA | 34 unique | 8/3 | cathode unspecified in certified local metadata | 4, 24, 43, 44 as recorded | six archive groups | 1.000 |
| CALCE | 9 | 9/9 | LCO (CS/CX), LFP (K2) | unavailable (`RT` label only) | one `RT` label across different subsets | 0.990 |
| HUST | 77 | 77/77 | LFP | 30 | heterogeneous multistage discharge; no usable within-source group split | 1.000 |

At cycle 100, NASA's 34 unique cells partition into 8 eligible, 11 with the three-observed-record EOL at or before the landmark, and 15 without follow-up after it. The corresponding exclusions are XJTU: 16 with too few early observations; TJU: 11 without post-landmark follow-up. MIT, CALCE and HUST have no cycle-100 exclusions under the audit rules. Among the 376 eligible cells, 228 have observed EOL and 148 are right-censored. The new main-text cohort figure is `figures_v4/fig_v4_source_cohort.pdf` and shows the same exclusive categories.

## S2. Benchmark and paired controls

`benchmark_metrics.csv` contains cycle-50/100 five-model scores, Harrell C, event-only MedAPE and supported-horizon counts. `benchmark_predictions.csv` gives one prediction per eligible target cell and model. Main-text Table 1 gives cycle-100 values. NASA has only the cycle-125 supported horizon, so integrated Brier is not estimable; its Harrell C and MedAPE are descriptive values from eight cells and three events.

`evidence08_metrics.csv`, `evidence08_predictions.csv`, and `evidence08_sensitivity.csv` give the within-source AFT and event-only regression comparisons. The within-source arm uses labels from other target cells and a different training size from the zero-shot arm. Event-only regression borrows the source-selected AFT scale to form a lognormal probability proxy; its Brier value is conditional on that choice. Ten alternative fold partitions and twenty paired seeds reuse the same cells and do not increase independent sample size. XJTU/TJU protocol-group results and group counts are in `evidence08p_groups.csv`; TJU group membership is also a chemistry distinction.

## S3. Endpoint, source correction and risk calibration

`endpoint_common_cohort.csv` identifies the common cohort across one-, three-, and five-observed-record endpoint definitions. `endpoint_model_metrics.csv` reports the fixed-parameter Cox and XGBoost AFT refits on common cells and within-source common scoring horizons. For XJTU Cox at cycle 100, the three-record and five-record scores are 0.0823095 and 0.0708729, respectively (displayed as 0.082 and 0.071). Different endpoint rules define different outcomes; a lower value does not validate one rule as more physical. The strict adjacent-integer-cycle rule has not been refitted in this manuscript.

`source_correction_metrics.csv` and `source_correction_cohort.csv` compare verified and historical source training on identical verified test labels and cells. The cycle-100 XGBoost AFT score improves on HUST (0.132 to 0.117) and worsens on CALCE (0.194 to 0.254). The risk-audit file `benchmark_horizons.csv` gives, among other points, cycle-500 CALCE predicted/held-out KM risk 0.672/0.111 and cycle-2000 HUST 0.451/0.896. These are fold-level risks, not individual calibration curves.

## S4. Target-label adaptation and conformal count

`recalibration_scores.csv`, `recalibration_horizons.csv`, `seed_target_recalibration_scores.csv`, and split-membership outputs preserve each draw and paired score. Calibration draws are random and unstratified. At the cycle-100 20% budget, calibration counts are MIT 25, XJTU 8, TJU 24, NASA 2, CALCE 2, HUST 16. Across 20 training seeds and 30 overlapping draws, mean shifted-minus-baseline integrated Brier differences are +0.0031, +0.0076, +0.0040, unavailable, -0.0757, and +0.0079 in that order. These draws are not 600 independent experiments per source.

`calibration_budget_feasibility.csv` and `protocol_v4.json` document the separate event-only absolute-residual finite-quantile count calculation. A conventional empirical 95% split-conformal quantile requires at least 19 calibration event residuals. Under random 20% target-cell allocation, the chance of reaching that count is 5.22% for TJU. This does not estimate or guarantee coverage for censored target cells.

## S5. Data access and declarations

The source repository and paper references appear in the main text. `v4_artifact_manifest.csv` records hashes for the internal analysis package. Source-specific redistribution permissions for MIT, NASA and CALCE, and the experiment-specific CALCE publication, remain under review. A public code/data repository and DOI have not yet been assigned. Real author names, contributions, funding and competing-interest statements require author confirmation before submission.

## S6. Graphical abstract

Use `figures_v4/graphical_abstract_v4.pdf` or `.png` if a graphical abstract is submitted. The v3 graphical abstract refers to an older analysis and must not accompany this manuscript.

