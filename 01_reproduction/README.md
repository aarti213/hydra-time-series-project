# 01 — Reproduction of HYDRA

**Authors:** Mahak Joshi (lead) and Shreed · COMP41850 AI for Time Series, Hydra reproduction and improvement project

Notebooks: `01_hydra_baseline.ipynb` and `03_reproduction_completion.ipynb` were written by Shreed; `02_hydra_ablation.ipynb` was written by Mahak.

This folder contains the reproduction part of the project: does the released code of *Hydra: Competing convolutional kernels for fast and accurate time series classification* (Dempster, Schmidt & Webb, DMKD 2023) give the published results, and do the paper's claims hold?

## What we reproduced

We ran the authors' official `hydra-authors-repo/code/hydra.py` **unmodified** (imported by file path) with the default k = 8 kernels per group and g = 64 groups, followed by `SparseScaler` and `RidgeClassifierCV` with the authors' ridge grid `np.logspace(-3, 3, 10)`. The 112 UCR datasets from the authors' results file (`hydra-authors-repo/results/results_ucr112_hydra.csv`) were loaded with `aeon.datasets.load_classification`. We compared our runs with the authors' published per-dataset results and with published results for other classifiers, ran the Rocket family next to Hydra on the same splits, repeated the design-choice ablations, and resampled the datasets where our default-split result was far from the paper.

## Claims

| Claim | Our evidence | Verdict |
|---|---|---|
| C1: Hydra beats TDE and MrSQM | From the published 30-resample results: +1.26 pp vs TDE (69/6/37) and +1.07 pp vs MrSQM (72/7/33), both Holm-significant (p = 0.003 and 0.004) | Holds |
| C2: Hydra ties ROCKET, TS-CHIEF and InceptionTime, and trails HC2 | No significant difference from ROCKET, TS-CHIEF or InceptionTime (Holm p = 0.31, 1.00, 1.00); −1.80 pp vs HC2 (Holm p < 0.001) | Holds |
| C3: Hydra improves MiniRocket and MultiRocket | 112 default splits: Hydra+MiniRocket − MiniRocket = +0.58 pp (Holm p < 0.001); Hydra+MultiRocket − MultiRocket = +0.25 pp (Holm p = 0.025); Hydra+MultiRocket is the best of the five models (0.8690) | Holds (small for MultiRocket) |
| C4: the default design choices are best | k: flat from 4 to 64; k = 1 is −15.68 pp, and clipping at k = 1 recovers +13.87 pp; max/hard only is −1.14 pp (Holm 0.010); max vs max+min −0.39 pp, soft+hard ≈ soft, no-diff −0.09 pp (not significant) | Partly holds |
| C5: the released code gives the published accuracies | Default split, seed 0: mean 0.8577 vs the paper's 0.8735; 18 datasets differ by more than 5 pp. Over 30 resamples of those 18 the gap becomes +0.27 pp, and the paper lies inside our 95% CI on 17/18. The default split is the worst of the 30 splits on 14/18. On the 18 outliers, the published Hydra run shows the same default-split penalty as ours (its resample 0 − its 30-resample mean: −8.76 pp; our resample 0 − the paper: −8.91 pp) | Holds once splits match |

All numbers come from `results/repro_partC_claims.csv` (C1, C2), `results/repro_partB_pairs.csv` (C3), `results/repro_partA_kg.csv`, `results/repro_partA_claims.csv` and the Part A tables in notebook 03 (C4), and `results/repro_partD_summary.csv`, `results/repro_partC_outliers.csv` and `results/hydra_default_baseline.csv` (C5). W/D/L counts are Hydra's wins/draws/losses, with a draw meaning |Δ| ≤ 0.05 pp.

## Folder contents

```
01_reproduction/
├── README.md
├── notebooks/
│   ├── 01_hydra_baseline.ipynb
│   ├── 02_hydra_ablation.ipynb
│   └── 03_reproduction_completion.ipynb
└── results/
    ├── *.csv, environment.json   (written by the notebooks, see below)
    └── figures/                  (*.png written by notebook 03)
```

## Notebooks

| Notebook | Parts / claims | Datasets and protocol | Main outputs (`results/`) | Run time (stated) |
|---|---|---|---|---|
| `01_hydra_baseline.ipynb` (Shreed) | Baseline for C5; input to notebook 03 and to `02_detach_hydra/` | Official Hydra, k = 8, g = 64, on the 112 UCR datasets, default split, seed 0; comparison with `results_ucr112_hydra.csv`. k/g sensitivity (k4/g128, k8/g64, k16/g32) on Coffee, ItalyPowerDemand and Wafer, seeds 0–9 | `hydra_default_baseline.csv`, `hydra_reference_comparison.csv`, `hydra_kg_sensitivity.csv`, `hydra_kg_summary.csv`, `hydra_baseline_initial_datasets.csv`, `ucr112_dataset_manifest.csv`, `ucr112_dataset_failures.csv`, `environment.json` | not stated |
| `02_hydra_ablation.ipynb` (Mahak) | First ablations of the design choices (C4, preliminary) | Max-only vs max+min (Coffee, seed 0); hard vs soft vs combined counting (Coffee, ECG200, GunPoint, seed 0); k/g at k × g = 512 (Coffee, ItalyPowerDemand, Wafer, seeds 0–2). Uses `RidgeClassifierCV()` with sklearn's default alphas | `ablation_max_min.csv`, `ablation_counting_methods_v2.csv`, `ablation_kg_sensitivity.csv` | not stated |
| `03_reproduction_completion.ipynb` (Shreed) | Part C: C1, C2, C5 · Part D: C5 · Part B: C3 · Part A: C4 · final claims summary | C: published results via `aeon.benchmarking.results_loaders` (no model runs). D: 30 resamples of the 18 outlier datasets, normalisation scan of the 112 datasets, EOG z-normalisation diagnostic. B: Hydra, MiniRocket, MultiRocket, Hydra+MultiRocket, Hydra+MiniRocket on the 112 datasets, default split. A: counting, differencing, clipping and k/g ablations on the 40 development datasets × 5 seeds. All models use the authors' ridge grid | `repro_partC_*.csv`, `repro_partD_*.csv`, `repro_partB_*.csv`, `repro_partA_*.csv`, `repro_claims_summary.csv`; `figures/repro_part{A,B,C,D}_*.png` | Part D 85.3 min, Part B 34.9 min, Part A 293.6 min (10 threads); Part C minutes |

Notebook 03 re-implements Mahak's ablations from notebook 02 as column selections of one convolution pass, and keeps her exact soft-min variant as `soft_only_mahak`. Its Part A therefore supersedes the preliminary results of notebook 02.

## How to reproduce

### 1. Environment

The results were produced on an Apple Silicon Mac (`results/environment.json`) with:

- Python 3.12
- numpy 2.3.5
- pandas 3.0.6
- scikit-learn 1.9.1
- torch 2.15.0
- aeon 1.6.0

From the repository root:

```bash
pip install -r requirements.txt
```

UCR datasets are downloaded automatically by `aeon.datasets.load_classification` into `data/ucr_tsml/` at the repository root and reused on later runs. Nothing in `data/` needs to be committed. Part C of notebook 03 also downloads published results from timeseriesclassification.com. If that fails, it reads `{Name}_accuracy.csv` files from `data/published_results/` and prints which files to download.

### 2. Paths

Every notebook starts with a path cell that finds the repository root (the folder containing `hydra-authors-repo/`). The notebooks therefore run from any working directory. They read:

- `hydra-authors-repo/code/hydra.py` (the official code);
- `hydra-authors-repo/results/*.csv` (the authors' published results);
- `01_reproduction/results/hydra_default_baseline.csv` (notebook 03 only).

They write to `01_reproduction/results/` and `01_reproduction/results/figures/`.

### 3. Run order and settings

```
01_hydra_baseline  →  03_reproduction_completion
02_hydra_ablation     (independent; reads no other notebook's output)
```

- `01_hydra_baseline.ipynb` must run before notebook 03, which asserts that `hydra_default_baseline.csv` covers all 112 datasets. It must also run before `02_detach_hydra/`, whose notebooks check ECG200 seed 0 = 0.8700 against the same file.
- Leave the saved settings: `RUN_ALL_112 = True` in notebook 01; `QUICK_RUN = False`, all `RUN_PART_*` and `RUN_CHECKS = True`, `INCLUDE_H_MINI = True`, `INCLUDE_SOFT_ONLY_MAHAK = True` and `N_SEEDS = 5` in notebook 03. `QUICK_RUN = True` is only a smoke test and writes `quick_*` files.
- Notebook 03 is meant to be run in the order C → D → B → A (its run guide), but top to bottom also works.

### 4. Things to know before re-running

- **Resuming (notebook 03).** Parts D, B and A resume from their CSVs. If `repro_partD_resamples.csv`, `repro_partB_rocket112.csv` and `repro_partA_ablations.csv` exist, completed units are skipped and only the analysis and figures are redone. **For a completely fresh run, move the existing `repro_*` CSVs out of `results/` first.**
- **Re-running notebook 02:** `02_hydra_ablation.ipynb` refuses to overwrite its results (cells 9, 15 and 18 raise `FileExistsError` if `ablation_max_min.csv`, `ablation_counting_methods_v2.csv` or `ablation_kg_sensitivity.csv` already exist in `01_reproduction/results/`). To re-run it, move those three CSVs out of `results/` first.
- **Indentation fix (notebook 02).** Cell 13 previously had `__init__` and `forward` outside the `HydraCountingVariant` class, so a fresh run stopped in cell 14 with a `TypeError`. The methods have been re-indented into the class (whitespace only). This restores the code that produced the saved outputs, and it now matches the copy in notebook 03 (cell 33).
- **Timings depend on the machine.** The timing columns are recomputed on every run, so their values will differ on other hardware, but the file names and accuracies will not.
- **Critical-difference figures.** `repro_partB_cd.png` and `repro_partC_cd_r0.png` are drawn only if aeon's `plot_critical_difference` works in your environment.
- **Published results.** Part C re-downloads the reference results on every run. If timeseriesclassification.com updates them, `repro_partC_*.csv` can change.
- **Memory.** The largest Part B datasets (ElectricDevices, Crop) need about 10–12 GB at the ridge fit.

### 5. Files restored from git

The results in `results/` were not regenerated in this folder. They are the files the notebooks wrote, copied byte for byte from commit `f30ee20` (Shreed, 2026-11-09, branch `shreed`, old folder `results/Shreed/`):

- notebook 01: `hydra_default_baseline.csv`, `hydra_reference_comparison.csv`, `hydra_kg_sensitivity.csv`, `hydra_kg_summary.csv`, `hydra_baseline_initial_datasets.csv`, `environment.json`; `ucr112_dataset_manifest.csv` and `ucr112_dataset_failures.csv` have the same content;
- notebook 03: all 16 `repro_*.csv` files and the 6 figures in `figures/`.

The notebook 02 CSVs are Mahak's own run (old folder `results/member3/`), which matches the numbers in that notebook.

## Correctness checks built into notebook 03

- ECG200, seed 0, with the authors' grid gives 0.8700, the same as notebook 01.
- Resample 0 (seed 0) reproduces notebook 01 exactly on all 18 outlier datasets.
- Standalone Hydra in Part B equals notebook 01 on all 112 datasets.
- The ablation `default` configuration equals the official Hydra output (`torch.equal`, ECG200 and Beef, seeds 0 and 1), and notebook 02's `HydraMaxOnly` and `HydraCountingVariant` equal the matching column selections.

All of these checks passed in the saved run.

## Limitations

- **C3 and C4 use the default split only** (1 split). The ablations use the 40 development datasets × 5 seeds, against the paper's 10 runs.
- **Only the 18 outlier datasets were resampled 30 times.** Our resamples are drawn with our own stratified sampler, not the authors' Java tsml splits, so agreement is expected within sampling noise, not number for number.
- **The authors' α grid often hits an endpoint** (for example 85% of the Part B fits). It was kept for fidelity to the paper.
- **Speed was not re-timed under the paper's conditions** (single core, 30 resamples).
- **Notebook 02 is preliminary:** few datasets and seeds, and sklearn's default ridge alphas.

## Acknowledgement

Claude (Anthropic) was used to help with language editing and with drafting and debugging the experiment notebooks. All code was reviewed and executed by the authors, and all reported results were produced by that code.
