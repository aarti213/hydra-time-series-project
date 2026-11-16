# Reproducing and improving HYDRA for time series classification

**COMP41850 AI for Time Series, University College Dublin**
**Team:** Mahak Joshi, Shreed Raskar, Aarti Bhosale

This repository reproduces the paper

> A. Dempster, D. F. Schmidt and G. I. Webb, "Hydra: Competing convolutional kernels for fast and accurate time series classification," *Data Mining and Knowledge Discovery*, vol. 37, pp. 1779–1805, 2023. Official code: <https://github.com/angus924/hydra>

checks whether its claims hold on the 112 UCR datasets, and proposes two improvements. Every number in the report comes from a notebook in this repository, and the result files and figures those notebooks produced are committed next to them.

📄 **Report:** [`report/Hydra_Reproduction_and_Improvement_Report.pdf`](report/Hydra_Reproduction_and_Improvement_Report.pdf)

---

## Repository layout

```
hydra-time-series-project/
├── README.md                  ← this file
├── requirements.txt           ← pinned package versions
├── report/                    ← final report (PDF)
├── hydra-authors-repo/        ← official Hydra code (code/hydra.py) and published results (results/*.csv), unmodified, GPL-3.0
├── 01_reproduction/           ← does the released code give the published results, and do the claims hold?   (Mahak, Shreed)
├── 02_detach_hydra/           ← improvement 1: pruning a trained Hydra by removing whole kernel groups        (Shreed)
└── 03_quant_hydra/            ← improvement 2: combining QUANT with Hydra, plus reproduction under the paper's protocol (Aarti)
```

Each work-package folder has its own `README.md` with full details, its notebooks in `notebooks/`, and its outputs in `results/`.

| Folder | Notebooks | What it answers |
|---|---|---|
| [`01_reproduction/`](01_reproduction/) | `01_hydra_baseline`, `02_hydra_ablation`, `03_reproduction_completion` | Claims C1–C5: accuracy of the released code, comparison with other classifiers, Hydra inside MiniRocket/MultiRocket, design-choice ablations |
| [`02_detach_hydra/`](02_detach_hydra/) | `01_pilot`, `02_development_run`, `03_heldout_confirmation`, `04_followup_fix_pilot`, `05_followup_fix_dev` | Can a trained Hydra be pruned to a fraction of its kernels and beat a Hydra trained at that size from scratch? |
| [`03_quant_hydra/`](03_quant_hydra/) | `01_hydra_reproduction_resamples`, `02_hydra_alpha_grid`, `03_multirocket_hydra_comparison`, `04_quant_hydra_combinations` | Do the results hold under the paper's 30-resample protocol? Does the ridge α grid matter? Does combining QUANT with Hydra help? |

---

## Headline results

**Reproduction** (`01_reproduction/`, `03_quant_hydra/` notebooks 01–03)
- **The released code reproduces the paper once the train/test splits match.**
  - On the default UCR split, the mean accuracy is 0.8577 against the paper's 0.8735. The gap comes from 18 datasets.
  - Over 30 resamples of those 18 datasets, the gap shrinks to +0.27 pp, and the paper lies inside our 95% CI on 17 of them.
  - Under the paper's protocol on 15 further datasets, 14/15 are within expected variance.
- **C1, C2 and C3 hold:**
  - Hydra beats TDE and MrSQM.
  - It ties ROCKET, TS-CHIEF and InceptionTime, and trails HC2.
  - It improves MiniRocket (+0.58 pp) and, slightly, MultiRocket (+0.25 pp).
- **C4 partly holds.** k = 8 is fine, but any k from 4 to 64 performs about the same, and clipping matters only at k = 1.
- **The ridge α grid isn't reported in the paper.** The README grid hits its upper bound in 37% of runs, but changing the grid barely moves mean accuracy.

**Detach-Hydra** (`02_detach_hydra/`; 72 held-out datasets, success rule fixed in advance)
- Keeping 1/16 of the kernel groups, Detach-Hydra is **+1.22 pp more accurate than a Hydra trained at that size** (49/4/19, Holm p = 0.0005).
- Its test-time transform is about **10× faster** than full Hydra.
- It costs 1.88 pp against full Hydra (88.44%) and about 3× full Hydra's training time.

**QUANT × Hydra** (`03_quant_hydra/` notebook 04; 18 datasets chosen in advance)
- QUANT with Hydra features beats QUANT by +3.5 pp (p = 0.006).
- Late fusion of QUANT with Hydra's dictionary-like features reaches 79.1% against Hydra's 75.5% (p = 0.15), at 0.72× Hydra's test time.

---

## How to reproduce

### 1. Set up the environment

```bash
git clone https://github.com/aarti213/hydra-time-series-project.git
cd hydra-time-series-project
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The results were produced on two machines with the same package versions:
- an Apple Silicon Mac (Python 3.12, used for `01_reproduction` and `02_detach_hydra`);
- a Windows laptop (Python 3.11, CPU-only torch, used for `03_quant_hydra`).

On Windows or Linux without a GPU, you can install the CPU build of torch first with `pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`. Each `results/` folder contains an `environment.json` recording the exact setup used.

### 2. Data

No data is stored in the repository. The UCR datasets are downloaded automatically by `aeon.datasets.load_classification` into `data/ucr_tsml/` (git-ignored) the first time a notebook needs them. An internet connection is needed for the first run.

### 3. Paths

Every notebook starts with the same path cell. It finds the repository root (the folder containing `hydra-authors-repo/`), loads the official `hydra.py` from `hydra-authors-repo/code/`, and writes only into its own folder's `results/`. The notebooks therefore run from any working directory.

### 4. Run order

| Step | Notebook | Needs | Approx. run time (stated in the notebooks) |
|---|---|---|---|
| 1 | `01_reproduction/notebooks/01_hydra_baseline.ipynb` | – | writes `hydra_default_baseline.csv`, which every `02_detach_hydra` notebook needs |
| 2 | `01_reproduction/notebooks/02_hydra_ablation.ipynb` | – | independent |
| 3 | `01_reproduction/notebooks/03_reproduction_completion.ipynb` | step 1 | about 7 h on 10 threads (Part D 85 min, Part B 35 min, Part A 294 min) |
| 4 | `02_detach_hydra/notebooks/01_pilot.ipynb` | step 1 | about 26 min, 1 thread |
| 5 | `02_detach_hydra/notebooks/02_development_run.ipynb` | step 1 | about 2.7 h, all cores |
| 6 | `02_detach_hydra/notebooks/03_heldout_confirmation.ipynb` | step 1 | overnight |
| 7 | `02_detach_hydra/notebooks/04_followup_fix_pilot.ipynb` | steps 1, 5 | about 11–15 min |
| 8 | `02_detach_hydra/notebooks/05_followup_fix_dev.ipynb` | steps 1, 5 | about 4–5 h |
| 9 | `03_quant_hydra/notebooks/01_hydra_reproduction_resamples.ipynb` | – | about 30–60 min, 1 thread |
| 10 | `03_quant_hydra/notebooks/02_hydra_alpha_grid.ipynb` | step 9 | see folder README |
| 11 | `03_quant_hydra/notebooks/03_multirocket_hydra_comparison.ipynb` | – | see folder README |
| 12 | `03_quant_hydra/notebooks/04_quant_hydra_combinations.ipynb` | – | see folder README |

The three folders are otherwise independent. `03_quant_hydra/` can run on its own, and so can `01_reproduction/` steps 1–3.

### 5. Things to know before re-running

- **You don't need to re-run anything to check the report.** All result CSVs and figures are committed in each folder's `results/`, and every notebook has its saved outputs.
- **Several notebooks resume from their saved results.** If the CSVs or per-run files already exist, completed units are skipped and only the analysis and figures are redone. **For a completely fresh run, move the existing files out of that folder's `results/` first.**
- **`01_reproduction/notebooks/02_hydra_ablation.ipynb` refuses to overwrite its CSVs.** Move its three `ablation_*.csv` files out of `01_reproduction/results/` before re-running it.
- **Timings depend on the machine.** Accuracy results are seeded and repeat exactly, but timing columns and speed-ups will differ in value on other hardware.
- Each folder README lists that folder's specific checks and caveats.

---

## Which notebook produces which part of the report

| Report part | Source |
|---|---|
| Reproduction: claims table, default split vs paper, 30-resample check of the 18 outliers, Rocket-family comparison, ablations | `01_reproduction/` (notebooks 01–03) |
| Reproduction under the paper's protocol (15 datasets × 30 resamples), α-grid lesson, Hydra + MultiRocket check | `03_quant_hydra/` (notebooks 01–03) |
| Detach-Hydra method, results, Figs. and Tables | `02_detach_hydra/` (notebooks 01–03; 04–05 for the future-work sentence) |
| QUANT × Hydra improvement | `03_quant_hydra/` (notebook 04) |

---

## Contributions

| Member | Work package | Notebooks | Report sections |
|---|---|---|---|
| **Mahak Joshi** | Reproduction (lead) | `01_reproduction/02_hydra_ablation` | Introduction, Related Work, Reproduction |
| **Shreed Raskar** | Detach-Hydra; reproduction baseline and completion | `02_detach_hydra/` (all), `01_reproduction/01_hydra_baseline`, `01_reproduction/03_reproduction_completion` | Detach-Hydra |
| **Aarti Bhosale** | QUANT × Hydra; reproduction under the paper's protocol; early Rocket-family comparison | `03_quant_hydra/` (all) | QUANT × Hydra, reproduction under the paper's protocol |

The full history of each member's work, including earlier experiments, is preserved in the git history and the original branches (`mahak`, `shreed`, `aarti`).

---

## Acknowledgements

- The official Hydra implementation and published results in `hydra-authors-repo/` are by Dempster, Schmidt and Webb, distributed under the GPL-3.0 licence.
- UCR datasets: Dau et al., "The UCR time series archive," IEEE/CAA J. Autom. Sinica, 2019.
- Claude (Anthropic) was used to help with language editing and with drafting and debugging the experiment notebooks. All code was reviewed and executed by the authors, and all reported results were produced by that code.