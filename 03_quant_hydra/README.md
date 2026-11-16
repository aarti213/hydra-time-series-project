# 03 — QUANT × Hydra, and reproduction under the paper's protocol

**Author:** Aarti Bhosale · COMP41850 AI for Time Series, Hydra reproduction and improvement project

This folder has two parts:

- **(a) Reproduction checks that complement `01_reproduction/`.** Hydra is re-run under the paper's own protocol (30 resamples, single thread). The folder also tests the effect of the unreported ridge α grid and checks the paper's Hydra + MultiRocket combination claim.
- **(b) The improvement: combining QUANT with Hydra.** QUANT (Dempster et al., 2024) is an interval method: quantiles over fixed dyadic intervals, classified with ExtraTrees. This part combines QUANT with full Hydra and with each of Hydra's two halves. The halves are the dictionary-like hard counts (`H-dict`) and the ROCKET-like soft counts (`H-rocket`).

All notebooks use the authors' unmodified `hydra-authors-repo/code/hydra.py` (imported by file path) with k = 8, g = 64, `SparseScaler`, and the README ridge grid `np.logspace(-3, 3, 10)` unless stated otherwise. In every notebook, resample 0 is the default UCR split and resample *i* is aeon's `stratified_resample_data(random_state=i)`. Resample *i* is therefore the same split in all four notebooks.

## Notebooks

| Notebook | Question | Datasets and protocol | Main outputs (`results/`) | Run time (stated) |
|---|---|---|---|---|
| `01_hydra_reproduction_resamples.ipynb` | Do the published Hydra accuracies hold under the paper's protocol? | 15 datasets (6 easy project datasets + 9 harder ones) × 30 resamples; Hydra seed = resample id; 1 CPU thread | `reproduction_resamples/`: `hydra_resample_runs.csv`, `hydra_vs_paper_summary.csv`, `environment.json`, `plots/hydra_resamples_vs_paper.png` | about 30–60 min on a laptop, 1 thread |
| `02_hydra_alpha_grid.ipynb` | Does the ridge α grid matter? | Same 15 datasets × 30 resamples and the same Hydra features; three grids: sklearn default (0.1, 1, 10), README (10⁻³–10³, 10 values), extended (10⁻³–2.2·10⁶, 15 values) | `alpha_grid/`: `runs/`, `alpha_grid_runs.csv`, `alpha_grid_summary.csv`, `environment.json`, `plots/alpha_grid_effect.png` | not stated (6 workers × 1 thread) |
| `03_multirocket_hydra_comparison.ipynb` | Does the paper's Hydra + MultiRocket result reproduce, and does Hydra add to MultiRocket? | 18 datasets × 10 resamples; MiniRocket, MultiRocket, Hydra, Hydra+MiniRocket, Hydra+MultiRocket; each block scaled as in the reference pipelines; isolated single-thread timing on resample 0 | `multirocket_hydra/`: `runs/`, `timing/`, `mrh_runs.csv`, `mrh_timing.csv`, `mrh_vs_paper.csv`, `mrh_paired_tests.csv`, `mrh_summary.csv`, `environment.json`, `plots/hydra_multirocket_gain.png`, `plots/accuracy_vs_train_time.png` | not stated (6 workers × 1 thread, then isolated timing) |
| `04_quant_hydra_combinations.ipynb` | Does combining QUANT with Hydra (or one Hydra half) improve accuracy, and at what cost? | The same 18 datasets × 10 resamples; 13 configurations: 4 baselines + early fusion (ridge or ExtraTrees) + late fusion; isolated single-thread timing on resample 0 | `quant_hydra/`: `runs/`, `diagnostics/`, `timing/`, `quant_hydra_runs.csv`, `quant_hydra_diagnostics.csv`, `quant_hydra_timing.csv`, `quant_hydra_vs_paper.csv`, `quant_hydra_paired_tests.csv`, `quant_hydra_summary.csv`, `environment.json`, `plots/gain_over_better_parent.png`, `plots/accuracy_vs_train_time.png`, `plots/complementarity.png` | about 2 h on a laptop (6 workers × 1 thread) |

The 18 datasets of notebooks 03 and 04 were chosen **before any run**, using the published 30-resample means of both papers:

- 6 **Hydra-favoured** datasets: PigAirwayPressure, PigCVP, OSULeaf, Rock, ToeSegmentation1, Wine;
- 6 **QUANT-favoured** datasets: SemgHandMovementCh2, EthanolLevel, SemgHandSubjectCh2, InlineSkate, ScreenType, Lightning2;
- 6 **neutral** datasets, with |published gap| < 0.5 pp: Herring, Adiac, Worms, DistalPhalanxTW, InsectWingbeatSound, WormsTwoClass.

Datasets with fewer than 50 test cases were skipped.

## Key results

### Reproduction under the paper's protocol (notebook 01)

- **Setup:**
  - unmodified `hydra.py`, k = 8, g = 64, `SparseScaler`, README α grid (10⁻³–10³, 10 values);
  - 15 datasets (6 easy, 9 harder) × 30 resamples (resample 0 = default split; seed = resample id);
  - 1 CPU thread.
- **14/15 datasets are within expected variance** (per-dataset z-test, |z| ≤ 1.96). The mean difference from the published means is **+0.06 pp** (mean |Δ| 0.44 pp), and a Wilcoxon signed-rank test finds no systematic difference (p = 0.64).
- **Only Herring is flagged** (+2.55 pp, z = 2.37, p ≈ 0.018). It is not significant after a Holm correction for 15 comparisons.
- **All 450 runs repeat exactly.** Notebook 02 re-ran them with the README grid on a different machine and PyTorch version: 450/450 runs gave identical accuracy, and 449/450 selected the identical α.
- **Run time:** single-thread training and test times are a median **1.38×** and **1.51×** the published values, a roughly constant ratio across datasets.
- **The default split alone is misleading.** With only resample 0, the mean |Δ| from the published means is **1.86 pp** (about 4× larger), up to +13.2 pp on Herring.

### Ridge α grid (notebook 02)

- **The README grid picks α = 1000** (its upper bound) in **37%** of runs, almost always on the harder datasets.
- **The sklearn default grid** (0.1, 1, 10) is worse: **−0.42 pp** vs the README grid (2 better / 4 same / 9 worse, Wilcoxon p = 0.040).
- **An extended grid up to 2·10⁶** removes the ceiling but gives **+0.04 pp** on average (p = 0.75). Single datasets move by up to ±3 pp: MiddlePhalanxTW **+3.16 pp**, Herring **−2.29 pp**.
- **The README grid is closest to the paper** (mean |Δ| 0.44 pp, vs 0.52 pp for the extended grid and 0.70 pp for the sklearn default).

### Hydra + MultiRocket (notebook 03)

- **Setup:** 18 datasets × 10 resamples. Each feature block is scaled as in the reference pipelines: `SparseScaler` on Hydra, `StandardScaler` on MiniRocket and MultiRocket.
- **Agreement with the paper:**
  - Hydra, Hydra+MiniRocket and Hydra+MultiRocket are **within expected variance of the paper on 18/18** datasets each.
  - QUANT is also within expected variance on 18/18 of these datasets (checked in notebook 04, `quant_hydra_vs_paper.csv`).
- **The gain of Hydra+MultiRocket over Hydra tracks the paper** dataset by dataset (Spearman ρ = 0.89): **+2.27 pp** vs **+2.20 pp** published on these 18 datasets.
- **Hydra+MultiRocket − MultiRocket = +0.78 pp** (14/18 datasets, p = 0.054), at **2.35×** MultiRocket's training time and **2.5×** its test time.

### QUANT × Hydra (notebook 04)

| Configuration | Mean accuracy (18 datasets) | Train × Hydra | Test × Hydra |
|---|---:|---:|---:|
| QUANT | 75.0% | 0.59 | 0.07 |
| Hydra | 75.5% | 1.00 | 1.00 |
| Q‖Hydra [ET] (early fusion, ExtraTrees) | 78.5% | 1.98 | 1.07 |
| Q+Hydra (late fusion) | 79.2% | 1.59 | 1.07 |
| **Q+H-dict** (late fusion, dictionary half) | **79.1%** | 1.23 | **0.72** |
| Q‖Hydra [R] (early fusion, one ridge) | 73.5% | 1.08 | 1.07 |

- **QUANT with Hydra features and ExtraTrees** (`Q‖Hydra [ET]`) beats QUANT by **+3.5 pp** (13/18 datasets, p = 0.006).
- **Late fusion of QUANT and the Hydra dictionary half** (`Q+H-dict`) averages the two models' class scores. It reaches **79.1% vs Hydra's 75.5%** (+3.5 pp, 11/18 datasets, p = 0.15), at **0.72×** Hydra's test time.
- **`Q+H-dict` is +1.3 pp vs Hydra+MultiRocket** (p = 0.15), at **0.41×** its test time. Both notebooks use the same splits and Hydra kernels; Hydra's accuracy is identical in both.
- **Concatenating everything into one ridge model fails.** `Q‖Hydra [R]` is **−6.8 pp vs the better parent** (p = 0.002), because the many standardised QUANT features dominate the ridge penalty.
- **Against the better of its two parents on each dataset, no combination wins overall** (Q+Hydra −1.1 pp, 4/14 datasets, p = 0.054). The average gain comes from robustness: each model fails on the other's favoured datasets, and the combination avoids the worst failures.

The p-value for `Q+H-dict` vs Hydra is computed from `quant_hydra_summary.csv` with the notebook's paired Wilcoxon test. `quant_hydra_paired_tests.csv` compares each combination with its own Hydra part, QUANT and the better parent.

## How to reproduce

### 1. Environment

The results were produced on Windows (`Windows-10-10.0.26200`) with 1 thread per task:

- Python 3.11 (3.11.17 for notebook 01, 3.11.0 for notebooks 02–04)
- torch 2.14.1+cpu (notebook 01) and 2.15.0+cpu (notebooks 02–04)
- numpy 2.3.5
- pandas 3.0.6
- scikit-learn 1.9.1
- aeon 1.6.0

Notebook 01 ran on a different machine from notebooks 02–04, which used an 18-thread CPU with 6 parallel workers. See the `environment.json` file in each `results/` subfolder.

From the repository root:

```bash
pip install -r requirements.txt
```

UCR datasets are downloaded automatically by `aeon.datasets.load_classification` into `data/ucr_tsml/` at the repository root. That folder is git-ignored, so no data needs to be committed.

### 2. Paths

Each notebook starts with a path cell that finds the repository root (the folder containing `hydra-authors-repo/`), so the notebooks run from any working directory. They read:

- `hydra-authors-repo/code/hydra.py` (the official code);
- `hydra-authors-repo/results/results_ucr112_hydra.csv` and `results_ucr112_variants.csv` (the authors' published results);
- `03_quant_hydra/reference_results/quant/accuracy_mean_ucr142.csv` (notebook 04).

The QUANT file holds QUANT's published accuracies, from github.com/angus924/quant. The local copy is byte-identical to the file in that repository, and the notebook only downloads it if the local copy is missing.

Each notebook writes to its own folder: `03_quant_hydra/results/reproduction_resamples/`, `alpha_grid/`, `multirocket_hydra/` or `quant_hydra/`. Figures go in that folder's `plots/` subfolder.

### 3. Run order and settings

```
01_hydra_reproduction_resamples  →  02_hydra_alpha_grid
03_multirocket_hydra_comparison      (independent)
04_quant_hydra_combinations          (independent)
```

- Notebook 02 reads `results/reproduction_resamples/hydra_resample_runs.csv` (written by notebook 01) to check that its README-grid runs match.
- Notebooks 03 and 04 read only the published results.
- Run each notebook top to bottom with the saved settings: `N_RESAMPLES = 30` (01, 02) or `10` (03, 04), `N_THREADS = 1` / one thread per task, `N_WORKERS = 6` (02–04), k = 8, g = 64.

### 4. Things to know before re-running

- **Resuming.** Every notebook skips work that is already saved:
  - notebook 01 appends each run to `hydra_resample_runs.csv` and skips (dataset, resample) pairs already in it;
  - notebooks 02–04 write one file per (dataset, resample) to `runs/` (and `diagnostics/` in 04) and skip complete files;
  - notebooks 03 and 04 also skip datasets whose file already exists in `timing/`.

  With the saved files present, a re-run only redoes the analysis, the summary CSVs and the figures. **For a completely fresh run, move these files out first:** `hydra_resample_runs.csv` (01), `runs/` (02), `runs/` and `timing/` (03), `runs/`, `diagnostics/` and `timing/` (04).
- **Overwritten on every run.** `environment.json`, the aggregated CSVs and the figures are rewritten each time, and `environment.json` then records the new machine. There are no overwrite guards.
- **Timings depend on the machine.** Cost is measured in isolated single-thread runs on resample 0 (03, 04). Timings from the parallel accuracy runs are dropped, because contention between workers inflates them. Ratios between models are meaningful; absolute seconds are not.
- **Matplotlib version.** The figure cell of notebook 02 uses `boxplot(..., orientation="horizontal")`, which needs matplotlib ≥ 3.10.
- **Notebook 03, cell 18** has no execution count in the saved file, but its saved outputs match `mrh_vs_paper.csv`.

### 5. Files restored from git

The four files in `results/reproduction_resamples/` were not in the copied folder. They were restored byte for byte from commit `25946ec` (Aarti, 2026-11-06, branch `aarti`, old folder `results/reproduction/`). They are unchanged in the latest commit on that branch (`40feba3`). The restored files are:

- `environment.json`;
- `hydra_resample_runs.csv`;
- `hydra_vs_paper_summary.csv`;
- `plots/hydra_resamples_vs_paper.png`.

All other files in `results/` and `reference_results/` are byte-identical to `origin/aarti` (`40feba3`, old folders `results/Aarti/` and `reference_results/quant/`).

## Correctness checks built into the notebooks

- With the README grid, notebook 02 reproduces all 450 runs of notebook 01 (identical accuracy, 449/450 identical α).
- `HydraPart` (one Hydra half computed on its own) equals the matching columns of the official Hydra output (`torch.equal`, notebook 04).
- Halves sliced from the official output and halves computed by their own transforms give identical accuracies for all 13 configurations (Wine, resample 1).
- Hydra's accuracy is identical in notebooks 03 and 04 (75.51%), which confirms the same splits and kernels.

## Folder layout

```
03_quant_hydra/
├── README.md
├── notebooks/
│   ├── 01_hydra_reproduction_resamples.ipynb
│   ├── 02_hydra_alpha_grid.ipynb
│   ├── 03_multirocket_hydra_comparison.ipynb
│   └── 04_quant_hydra_combinations.ipynb
├── reference_results/
│   └── quant/accuracy_mean_ucr142.csv     (published QUANT accuracies, 142 UCR datasets)
└── results/
    ├── reproduction_resamples/            (notebook 01: 3 files + plots/)
    ├── alpha_grid/                        (notebook 02: 3 files + plots/)
    │   └── runs/                          (450 CSVs, one per dataset × resample)
    ├── multirocket_hydra/                 (notebook 03: 6 files + plots/)
    │   ├── runs/                          (180 CSVs, one per dataset × resample)
    │   └── timing/                        (18 CSVs, one per dataset)
    └── quant_hydra/                       (notebook 04: 7 files + plots/)
        ├── runs/                          (180 CSVs, one per dataset × resample)
        ├── diagnostics/                   (180 CSVs: QUANT/Hydra agreement per run)
        └── timing/                        (36 CSVs: <dataset>.csv and <dataset>__blocks.csv)
```

## Limitations

- **Small dataset samples.** The notebooks use 15 and 18 datasets, chosen for these questions. The 18 datasets of notebooks 03 and 04 were picked for a QUANT-vs-Hydra contrast, so they are harder and more varied than an average UCR sample.
- **10 resamples in notebooks 03 and 04,** against the papers' 30. The comparison with the published means accounts for n = 10 vs 30.
- **The late-fusion gains are not significant** (p = 0.15 vs Hydra and vs Hydra+MultiRocket). With 6 datasets per group the smallest possible Wilcoxon p is 0.031, and many comparisons are made.
- **The QUANT reference CSV is published results,** not re-run by us. QUANT was re-run in notebook 04 only on the 18 selected datasets.
- **aeon's resamples follow the tsml procedure** but are not guaranteed to be the authors' exact splits, so the notebooks compare 30-resample distributions, not individual resamples.
- **"Better parent" is chosen per dataset after the fact,** which slightly favours the parents.

## Acknowledgement

Claude (Anthropic) was used to help with language editing and with drafting and debugging the experiment notebooks. All code was reviewed and executed by the author, and all reported results were produced by that code.
