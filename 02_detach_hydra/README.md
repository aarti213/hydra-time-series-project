# 02 — Detach-Hydra: importance-ranked group pruning for Hydra

**Author:** Shreed · COMP41850 AI for Time Series, Hydra reproduction and improvement project

This folder contains the improvement part of the project: **Detach-Hydra**, a way to make a trained Hydra classifier smaller and faster by removing whole groups of competing kernels in order of their importance to the classifier.

## Idea in one paragraph

Hydra arranges its random convolutional kernels in groups of 8 that compete only with each other. Removing a **whole group** therefore leaves every other feature exactly unchanged, and the removed group no longer needs to be computed. Detach-Hydra:

1. trains a full Hydra (64 groups per dilation);
2. scores each group by the L2 norm of its ridge coefficients;
3. removes the weakest 10% of groups and refits, repeating until 1/2, 1/4, 1/8 or 1/16 of the groups are left;
4. rebuilds a smaller Hydra (`PrunedHydra`) that contains only the kept groups.

The final version (**v2**) averages the group scores over 5 stratified 80% subsamples of the training data, which reduces overfitting of the ranking. The main baseline is a **small Hydra** trained from scratch with the same number of groups.

## Headline result (72 held-out UCR datasets, 5 splits each)

| Groups kept | Detach-Hydra v2 | Small Hydra | v2 − Small (pp) | W/D/L | Holm p |
|---|---|---|---|---|---|
| 1/2 | 88.23% | 88.10% | +0.13 | 38/8/26 | 0.29 |
| 1/4 | 87.68% | 87.38% | +0.31 | 39/8/25 | 0.29 |
| 1/8 | 87.20% | 86.63% | +0.56 | 42/7/23 | 0.048 |
| **1/16** | **86.56%** | **85.34%** | **+1.22** [95% CI +0.55, +1.97] | **49/4/19** | **0.0005** |

Full Hydra scores 88.44% on the same datasets.

- The success rule was fixed before the run: at a budget of 1/4 or smaller, a mean gain of at least +0.5 pp, wins at least twice the losses, and Holm-corrected p < 0.05. **It is met at 1/16.**
- At 1/16 the test-time transform is about **10× faster** than full Hydra (development run, single thread).
- Training costs about **3.2×** full Hydra per split (median).

## Folder contents

```
02_detach_hydra/
├── README.md
├── notebooks/
│   ├── 01_pilot.ipynb
│   ├── 02_development_run.ipynb
│   ├── 03_heldout_confirmation.ipynb
│   ├── 04_followup_fix_pilot.ipynb
│   └── 05_followup_fix_dev.ipynb
└── results/
    ├── *.csv          (one set of CSVs per notebook, see below)
    └── figures/       (*.png written by the notebooks)
```

## Notebooks

| Notebook | Stage | Datasets | What it does | Main outputs (`results/`) | Used in report | Run time (stated) |
|---|---|---|---|---|---|---|
| `01_pilot.ipynb` | Pilot | 9 development datasets × 3 seeds, default split | Implements group pruning and `PrunedHydra`, runs the equivalence tests, compares Detach with random pruning and small Hydra, and times the pruned transforms | `detach_hydra_pilot*.csv`, `figures/detach_hydra_*.png` | Table 1 (pilot row) | about 26 min, 1 thread |
| `02_development_run.ipynb` | Development run (v1) | 40 development datasets × 5 splits | Detach v1 vs small Hydra at 4 budgets, with the controls random, random-even and Detach-even; applies the success rule; times pruned models on 6 datasets | `detach_decisive_runs.csv`, `_paths`, `_kept_groups`, `_timing`; `figures/detach_decisive_*.png` | Table 1, Table 3, Table 4 (speed), Fig. 4 | about 2.7 h, all cores (timing on 1 thread) |
| `03_heldout_confirmation.ipynb` | Stage A (v1 vs v2) + held-out confirmation (v2) | Stage A: 6 development datasets; confirmation: 72 held-out datasets × 5 splits | Checks v2 against v1 on development data, then runs full Hydra, Detach v2 and small Hydra on the held-out datasets and applies the same success rule | `detach_confirm_runs.csv`, `_paths`, `_kept_groups`, `_devcheck_runs`; `figures/detach_confirm_*.png` | Table 1, Table 2, Table 4 (training cost), Figs. 1(c), 2, 3, 4 | overnight run |
| `04_followup_fix_pilot.ipynb` | Follow-up (not in main results) | development datasets | Pilot of two variants aimed at the rare large losses: more subsamples (Fix A), and half-ranked, half-even selection (Fix D) | `detach_fix_pilot*.csv`, figures | One future-work sentence in the report | about 11–15 min |
| `05_followup_fix_dev.ipynb` | Follow-up (not in main results) | 34 development datasets × 5 splits | Development run of Fix A and Fix D against v2 with a pre-registered selection rule | `detach_fix_dev_*.csv`, `figures/detach_fix_dev_*.png` | One future-work sentence in the report | about 4–5 h |

The CSV names starting with `detach_decisive_` belong to the development run. "Decisive" was the working name of that stage.

The 40 development datasets are the ones Hydra's authors used for development. The 72 held-out datasets are the rest of the 112 UCR datasets. They were used only once, after the method and the success rule had been frozen.

## How to reproduce

### 1. Environment

Results were produced on an Apple Silicon Mac with:

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

UCR datasets are downloaded automatically by `aeon.datasets.load_classification`, so no data needs to be committed.

### 2. Paths

Every notebook starts with a path cell that finds the repository root (the folder containing `hydra-authors-repo/`). The notebooks therefore run from any working directory, and they always write to `02_detach_hydra/results/` and `02_detach_hydra/results/figures/`. The official Hydra code is imported from `hydra-authors-repo/code/hydra.py`.

### 3. Prerequisite

All five notebooks run a sanity check against the reproduction baseline: ECG200, seed 0, accuracy 0.8700. This needs:

```
01_reproduction/results/hydra_default_baseline.csv
```

That file is produced by `01_reproduction/notebooks/01_hydra_baseline.ipynb`, so **run that notebook first**. If the reproduction folder stores the file elsewhere, update the `BASELINE_CSV` line in the path cell of each notebook.

### 4. Run order

```
01_reproduction/notebooks/01_hydra_baseline.ipynb   (prerequisite)
01_pilot → 02_development_run → 03_heldout_confirmation → 04_followup_fix_pilot → 05_followup_fix_dev
```

Run each notebook top to bottom with the saved settings (`QUICK_RUN = False`).

### 5. Things to know before re-running

- **Resuming.** Notebooks 03, 04 and 05 resume from their CSVs. If the CSVs already exist, completed dataset/split pairs are skipped and only the analysis and figures are redone. **For a completely fresh run, move the existing CSVs out of `results/` first.**
- **Timings depend on the machine.** Timing columns, and `detach_decisive_timing.csv`, are recomputed on every run. Their values will differ on other hardware, but the file names and accuracy results will not.
- **Exact-match checks.** Notebooks 04 and 05 contain a check that their v2 results match `detach_decisive_runs.csv` exactly (check 8 in 04, check 9 in 05). On a different machine or thread count, tiny numerical differences could make this check fail even though the results are equivalent.
- **Critical-difference figures.** The two CD figures in notebook 02 are drawn only if aeon's `plot_critical_difference` works in your environment.
- **Threads.** Notebook 01 and all speed measurements use 1 thread. The large runs (02–05) use all available cores.
- **Logs.** Background-run logs (`*_log.txt`) are not part of the repository.

## Correctness checks built into the notebooks

- `PrunedHydra` with all groups kept equals the official Hydra output (`torch.equal`).
- Every rebuilt pruned transform equals the kept columns of the full transform (`torch.equal`, checked on every dataset).
- Fitting `SparseScaler` once and then selecting columns equals refitting it (maximum difference 0).
- Every pruned model keeps exactly the budgeted number of groups, the same as small Hydra.
- The pruning path and the choice of ridge α use training data only.
- v2 with 1 subsample of the full training set gives the same selections as v1.

## Limitations

- **5 splits per dataset,** rather than the 30 resamples used by Hydra's authors.
- **Speed was timed in the development run only,** on 6 datasets with v1 selections.
- **v1 was not run on the held-out datasets,** so the benefit of v2 over v1 rests on the 6 Stage A datasets.
- **The ranking can still fail on noisy datasets with few classes.** Earthquakes and DistalPhalanxOutlineAgeGroup lose more than 5 pp at some budget.
- **Training costs about 3× full Hydra,** and much more on large training sets because of the repeated subsample fits.

## Acknowledgement

Claude (Anthropic) was used to help with language editing and with drafting and debugging the experiment notebooks. All code was reviewed and executed by the author, and all reported results were produced by that code.