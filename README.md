# Cross-dataset battery lifetime prediction (v4)

This repository contains the reproducibility materials for the v4 Applied Energy manuscript. The study evaluates landmarked survival models across six public battery sources with source-held-out testing, endpoint sensitivity, source-correction checks, and target-label recalibration.

## Contents

- `AE_paper_elsarticle_v4.tex`: canonical manuscript source; compile from the repository root.
- `paper/`: compiled v4 manuscript PDF and Supplementary file S1.
- `figures_v4/`: v4 figures and optional graphical abstract.
- `rebuild_v4/`: analysis and validation scripts.
- `data/derived/`: aggregate metrics, protocols, reference audit records, and hashes.

## Data access

Raw source files are not redistributed here. Obtain them from the providers and follow their current terms:

- MIT/Severson dataset: [Nature Energy article and DOI](https://doi.org/10.1038/s41560-019-0356-8).
- NASA PCoE battery data: [NASA Open Data catalog](https://data.nasa.gov/dataset/prognostics-in-battery-health-management).
- CALCE CS2/CX2/K2 data: [CALCE Battery Data Archive](https://web.calce.umd.edu/batteries/data/).
- XJTU, TJU, and HUST sources: see the source-specific citations and URLs in `AE_paper_elsarticle_v4.tex` and `paper/AE论文_Supplementary_v4.md`.

The provider pages are the authoritative access points. This repository does not grant or imply a license to redistribute third-party data.

## Reproduction limits

The scripts expect locally downloaded source files and the internal audit inputs described in their headers. The public release contains aggregate outputs and hashes, but not raw files or row-level prediction/member files. Results are retrospective and exploratory; NASA has one supported scoring horizon and CALCE has nine eligible cells at cycle 100.

## License

Original scripts and documentation in this repository are released under the MIT License. Third-party data, trademarks, and source-provider materials remain subject to their respective terms. The manuscript's author, funding, contribution, and repository DOI metadata must be finalized before submission.

## Version

v4 release candidate, 2026-09-30. This is a reproducibility release, not a claim that all source data may be redistributed.
