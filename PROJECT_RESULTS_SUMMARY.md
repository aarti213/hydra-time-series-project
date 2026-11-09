# HYDRA Time Series Classification: Project Results Summary

> **Purpose of this document:** a detailed, factual summary of every experiment in our three project notebooks. It is meant for an AI assistant to read so it can **find gaps, weaknesses and possible improvements** in our work. All numbers come from the saved notebook outputs and result CSVs (as of 2026-11-07). Notebooks 2 and 3 were **bug-fixed and re-run** on 2026-11-07; all numbers here are from the corrected runs (see §2 and §3 for what was fixed).

---

## 0. Project context

- **Goal:** study HYDRA (Dempster, Schmidt & Webb, 2023, *"Hydra: Competing Convolutional Kernels for Fast and Accurate Time Series Classification"*, DMKD 37:1779–1805) and **find improvements** to it.
- **Team / notebooks:**
  1. `notebooks/shreed_hydra_baseline.ipynb` (Shreed): reproduces the baseline on **all 112 UCR datasets** and runs a k/g sensitivity study. Results are in `results/Shreed/`.
  2. `notebooks/mahak_hydra_ablation.ipynb` (Mahak): ablations of max-only vs max+min and hard vs soft counting, plus k/g sensitivity. Results are in `results/Mahak/` (the older `results/member3/` files come from the buggy run and are obsolete).
  3. `notebooks/aarti_combined_experiments.ipynb` (Aarti): compares MiniRocket, MultiRocket and Hydra+MultiRocket, plus a repeated Wafer timing benchmark. Results are in `results/Aarti/`.
- **Shared pipeline (all notebooks):**
  - Hydra comes from the **authors' official code**, `hydra-authors-repo/code/hydra.py` (imported unchanged; variants are subclasses).
  - Data comes from the UCR archive via `aeon.datasets.load_classification`, using the **predefined default train/test split** only (no resampling).
  - Classifier: `SparseScaler` (authors' code) followed by `RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))`, now used consistently in all three notebooks.
  - Metrics: accuracy, macro F1, and train and test wall-clock time.
- **Hardware:** all three notebooks' current results were produced on the same machine: macOS on Apple Silicon (CPU only, PyTorch 2.15, aeon 1.6, sklearn 1.9.1, Python 3.12). **Timings are therefore comparable across notebooks.**

### How the official Hydra works (relevant to the ablations)

- Constructor: `Hydra(input_length, k=8, g=64, seed=None)`. There are **g groups of k kernels** (k×g = 512 kernels per dilation). Kernels have length 9, are drawn from a normal distribution and are mean-centred (`W = W - W.mean(-1)`).
- Dilations are exponential, `2^0 … 2^log2((L-1)/8)`. The transform runs on both **X and diff(X)** (`divisor = 2`, i.e. first-order differencing is ON). There is **no input normalisation** inside Hydra.
- `forward()`: at each time point, within each group, the kernel with the **max** response "wins" and the kernel with the **min** response "wins".
  - **Max channel:** `scatter_add` of the max *values*. This is **soft counting** (sums the activations).
  - **Min channel:** `scatter_add` of *ones*. This is **hard counting** (counts wins).
- This yields 2 × num_dilations × divisor × h × k features. For example: 6144 for length 286 and 8192 for length 1460.
- `SparseScaler`: computes `sqrt(clamp(X,0))`, then standardises, with an epsilon tied to the fraction of zeros (`exponent = 4`). It is fitted on training features only.
- The official repo also contains `softmax.py` (Hydra + SGD for >10k training examples) and `hydra_multivariate.py` (experimental). **Neither has been used by us.**

### Important fact about the paper's reference numbers

The authors' published results (`results_ucr112_hydra.csv`, `results_ucr112_variants.csv`) are the **mean accuracy over 30 resamples** of train/test, as stated in their README ("UCR Archive (112 Datasets, 30 Resamples)"). For example, the reference for Coffee is 0.997619, which is not achievable on a single 28-case test set. **All our runs use only the single default split**, so per-dataset comparisons against the paper are not like-for-like.

---

## 1. Notebook 1: `shreed_hydra_baseline.ipynb` (Shreed): baseline reproduction

### 1.1 Setup checks

- The official `hydra.py` was loaded successfully and the constructor and `forward()` source were inspected.
- The 112-dataset list comes from the authors' `results_ucr112_hydra.csv`, and all 112 names are unique.
- **All 112 datasets downloaded and validated with 0 failures.** Every dataset is univariate with equal-length series.
  - Series length: min 15, median 315, mean 551, max 2844.
  - Classes: 2 to 60. 40 datasets are binary, 33 have 3–5 classes, 19 have 6–10 and 20 have more than 10.
  - Largest training sets: ElectricDevices (8926), Crop (7200), FordB (3636), FordA (3601).
  - Smallest training sets: DiatomSizeReduction (16), InsectEPGSmallTrain (17), and several datasets with 20.
- Smoke test on Coffee (seed 0): accuracy 1.000, macro F1 1.000, 6144 features.
- Initial three datasets (seed 0): ACSF1 0.860, Adiac 0.818, ArrowHead 0.834.

### 1.2 Default Hydra on ALL 112 UCR datasets (k=8, g=64, seed=0, single default split)

`RUN_ALL_112 = True`: **112 succeeded, 0 failed.** Files: `results/Shreed/hydra_default_baseline.csv` and `results/Shreed/hydra_reference_comparison.csv`.

**Aggregate comparison with the paper (paper = mean of 30 resamples):**

| Metric | Ours (seed 0, default split) | Paper Hydra |
|---|--:|--:|
| Mean accuracy (112 datasets) | **0.8577** | 0.8735 |
| Median accuracy | 0.9109 | 0.9300 |
| Mean macro F1 | 0.8364 | – |
| Mean difference (ours − paper) | **−1.58 pp** | |
| Median difference | −0.07 pp | |
| Std of differences | 5.41 pp | |
| Better / tied (±0.05 pp) / worse than paper | 47 / 8 / 57 | |
| Datasets within ±1 pp / ±2 pp | 59 / 76 | |
| Datasets differing by more than 5 pp | 18 | |
| Wilcoxon signed-rank (ours vs paper) | p = 0.025 (paper significantly higher) | |
| Datasets with 100% accuracy / ≥99% / <80% | 12 / 17 / 30 | |
| Total train time / test time, all 112 | 317.7 s / 554.6 s | 895.9 s / 1565.3 s (authors' hardware) |

For context, the paper's reported means for its combinations over the 112 datasets are:

| Model | Mean accuracy |
|---|--:|
| Hydra | 0.8735 |
| Hydra+MiniRocket | 0.8815 |
| Hydra+MultiRocket | **0.8839** |
| Hydra+Rocket | 0.8802 |

**Largest drops vs paper (ours − paper):**

| Dataset | n_train | n_test | Length | Classes | Ours | Paper | Diff (pp) |
|---|--:|--:|--:|--:|--:|--:|--:|
| EOGVerticalSignal | 362 | 362 | 1250 | 12 | 0.4945 | 0.7640 | −26.95 |
| EOGHorizontalSignal | 362 | 362 | 1250 | 12 | 0.5718 | 0.8270 | −25.52 |
| RefrigerationDevices | 375 | 375 | 720 | 3 | 0.5067 | 0.7561 | −24.94 |
| ScreenType | 375 | 375 | 720 | 3 | 0.4907 | 0.6241 | −13.34 |
| ElectricDevices | 8926 | 7711 | 96 | 7 | 0.7531 | 0.8817 | −12.86 |
| Computers | 250 | 250 | 720 | 2 | 0.7040 | 0.8185 | −11.45 |
| FordB | 3636 | 810 | 500 | 2 | 0.8247 | 0.9322 | −10.75 |
| Ham | 109 | 105 | 431 | 2 | 0.7429 | 0.8476 | −10.48 |
| FaceAll | 560 | 1690 | 131 | 14 | 0.8988 | 0.9843 | −8.55 |
| LargeKitchenAppliances | 375 | 375 | 720 | 3 | 0.8533 | 0.9356 | −8.22 |
| EthanolLevel | 504 | 500 | 1751 | 4 | 0.5860 | 0.6643 | −7.83 |
| Meat | 60 | 60 | 448 | 3 | 0.9167 | 0.9839 | −6.72 |

**Largest gains vs paper:**

| Dataset | Ours | Paper | Diff (pp) |
|---|--:|--:|--:|
| Herring | 0.7500 | 0.6182 | +13.18 |
| PigCVP | 0.9663 | 0.9111 | +5.53 |
| ACSF1 | 0.8600 | 0.8067 | +5.33 |
| Lightning7 | 0.8219 | 0.7831 | +3.88 |
| PigArtPressure | 0.9856 | 0.9516 | +3.40 |
| PigAirwayPressure | 0.7692 | 0.7404 | +2.88 |
| Rock | 0.9200 | 0.8920 | +2.80 |
| Beef | 0.8333 | 0.8056 | +2.78 |

**Hardest datasets for Hydra (lowest accuracy in our run).** These are the most room for improvement:

| Dataset | Length | Classes | n_train | Ours | Paper |
|---|--:|--:|--:|--:|--:|
| Phoneme | 1024 | 39 | 214 | 0.3238 | 0.3297 |
| ScreenType | 720 | 3 | 375 | 0.4907 | 0.6241 |
| InlineSkate | 1882 | 7 | 100 | 0.4909 | 0.4959 |
| EOGVerticalSignal | 1250 | 12 | 362 | 0.4945 | 0.7640 |
| SemgHandMovementCh2 | 1500 | 6 | 450 | 0.4956 | 0.4927 |
| RefrigerationDevices | 720 | 3 | 375 | 0.5067 | 0.7561 |
| Haptics | 1092 | 5 | 155 | 0.5130 | 0.5164 |
| MiddlePhalanxTW | 80 | 6 | 399 | 0.5325 | 0.5578 |
| EOGHorizontalSignal | 1250 | 12 | 362 | 0.5718 | 0.8270 |
| EthanolLevel | 1751 | 4 | 504 | 0.5860 | 0.6643 |
| MiddlePhalanxOutlineAgeGroup | 80 | 3 | 400 | 0.5974 | 0.6632 |
| InsectWingbeatSound | 256 | 11 | 220 | 0.6545 | 0.6600 |
| Computers | 720 | 2 | 250 | 0.7040 | 0.8185 |
| Crop | 46 | 24 | 7200 | 0.7336 | 0.7348 |
| WordSynonyms | 270 | 25 | 267 | 0.7429 | 0.7499 |

**Breakdown by dataset characteristics (mean accuracy, ours vs paper):**

| Group | n | Ours | Paper | Diff (pp) |
|---|--:|--:|--:|--:|
| Length ≤100 | 22 | 0.8360 | 0.8529 | −1.69 |
| Length 101–300 | 30 | 0.9223 | 0.9266 | −0.43 |
| Length 301–700 | 29 | 0.8922 | 0.8994 | −0.72 |
| **Length >700** | 31 | **0.7783** | 0.8123 | **−3.40** |
| 2 classes | 40 | 0.9075 | 0.9190 | −1.15 |
| 3–5 classes | 33 | 0.8550 | 0.8775 | −2.25 |
| 6–10 classes | 19 | 0.8177 | 0.8203 | −0.25 |
| >10 classes | 20 | 0.8005 | 0.8263 | −2.57 |
| n_train ≤50 | 27 | 0.9486 | 0.9527 | −0.41 |
| n_train 51–200 | 31 | 0.8867 | 0.8854 | +0.13 |
| n_train 201–1000 | 47 | 0.7864 | 0.8173 | −3.09 |
| n_train >1000 | 7 | 0.8576 | 0.8918 | −3.42 |

**Observations and possible explanations (not yet verified):**

- **EOGVertical and EOGHorizontal are not z-normalised** in the archive. Their training series have a per-series mean of about 4.2 and std of about 91.5. Every other dataset we checked (Refrigeration, Computers, Coffee, GunPoint, FordB, Ham) has a mean of about 0 and std of about 1. Hydra does not normalise its input, so these large drops may be caused by the missing normalisation. They could also come from the authors' preprocessing differing from ours.
- The **720-length "electric devices" family** (Refrigeration, ScreenType, Computers, LargeKitchen, SmallKitchen), **ElectricDevices** and **FordB** are known to differ strongly between the default split and resampled splits. The paper's 30-resample average may hide this.
- Overall: the **median** difference is about 0, but the **mean** is pulled down by a handful of datasets. Most of the gap is probably a protocol mismatch (one split and one seed vs 30 resamples), not a bug.
- Hydra is weakest on **long series (>700)**, on **many-class** problems and on **mid-to-large training sets**. These are natural targets for improvement.
- Speed: the slowest datasets were ElectricDevices (36.2 s train), HandOutlines (31.6 s), FordA and FordB (about 26 s each) and StarLightCurves (85 s test). Our runs were about 2.8× faster than the authors' reported times (different hardware).

The per-dataset table for all 112 datasets is in the **Appendix**.

### 1.3 k/g sensitivity (k×g fixed at 512; 10 seeds; Coffee, ItalyPowerDemand and Wafer; 90 runs, 0 failures)

Files: `results/Shreed/hydra_kg_sensitivity.csv` and `results/Shreed/hydra_kg_summary.csv`.

| Dataset | Config | Mean acc | Std | Mean train s | Mean test s |
|---|---|--:|--:|--:|--:|
| Coffee | k4_g128 / k8_g64 / k16_g32 | 1.0000 (all) | 0 | 0.14 / 0.14 / 0.14 | 0.13 / 0.14 / 0.13 |
| ItalyPowerDemand | k4_g128 | **0.9674** | 0.0027 | 0.029 | 0.279 |
| ItalyPowerDemand | k8_g64 (default) | 0.9673 | 0.0023 | 0.030 | 0.272 |
| ItalyPowerDemand | k16_g32 | 0.9651 | 0.0023 | 0.029 | 0.271 |
| Wafer | k4_g128 | 1.0000 | 0.0001 | 1.93 | 10.54 |
| Wafer | k8_g64 | 1.0000 | 0 | 1.81 | 9.79 |
| Wafer | k16_g32 | 1.0000 | 0 | 1.76 | 9.50 |

**Conclusion:** accuracy is insensitive to k/g on these three datasets. However, **these three datasets are near ceiling** (100% or 96–97%), so the experiment cannot reveal differences. On Wafer, k16_g32 is slightly faster.

---

## 2. Notebook 2: `mahak_hydra_ablation.ipynb` (Mahak): ablations (corrected and re-run)

**Status:** fixed and re-run on 2026-11-07 on the same Mac as Notebook 1. Two bugs were fixed before the re-run:

1. The `HydraCountingVariant` class had `__init__` and `forward` indented outside the class, so the counting variants were not actually being applied as intended.
2. The classifier used `RidgeClassifierCV()` with default alphas `(0.1, 1, 10)`. It now uses `RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))`, the same as the authors and the other notebooks.

After the fix, a check confirmed that `HydraCountingVariant(max="soft", min="hard")` produces **exactly the same features as the original Hydra**. Results are now in `results/Mahak/`; the older `results/member3/` files are from the buggy run and are obsolete.

The notebook's planned outline:

1. Setup
2. Baseline verification
3. Exp 1: max-only vs max+min
4. Exp 2: hard vs soft vs combined counting
5. Exp 3: first-order differencing on/off
6. Exp 4: clipping on/off
7. Exp 5: sensitivity analysis
8. Results summary and visualisations
9. Discussion

**Only Exp 1, Exp 2 and a k/g sensitivity study have been done.** Differencing on/off, clipping on/off and the visualisations are still **not done**.

Common settings: k=8, g=64 unless stated otherwise. `runtime_seconds` covers the train and test transforms, scaling, fitting and prediction.

### 2.1 Experiment 1: max-only vs max+min (Coffee only, seed 0)

`HydraMaxOnly` keeps only the max channel (soft counting: `scatter_add` of the max values) and drops the min channel.

| Variant | Accuracy | Macro F1 | Features | Runtime (s) |
|---|--:|--:|--:|--:|
| Original (max+min) | 1.000 | 1.000 | 6144 | 0.282 |
| Max-only | 1.000 | 1.000 | 3072 | 0.225 |

**Conclusion:** no accuracy loss, half the features and about 20% faster. However, there is **one dataset, one seed, and the dataset is at ceiling**, so this cannot show whether the min channel helps. The notebook defines a 3-dataset list (Coffee, ECG200, GunPoint), but only Coffee was used in Exp 1.

### 2.2 Experiment 2: hard vs soft vs combined counting (Coffee, ECG200, GunPoint; seed 0)

`HydraCountingVariant(max_counting, min_counting)`: soft adds the activation value; hard adds 1.

| Variant | Max channel | Min channel |
|---|---|---|
| `hard_only` | hard | hard |
| `soft_only` | soft | soft |
| `combined_default` | soft | hard (**= the original Hydra**) |

| Dataset | hard_only | soft_only | combined_default (original) |
|---|--:|--:|--:|
| Coffee | 1.00 (F1 1.000) | 1.00 (1.000) | 1.00 (1.000) |
| ECG200 | **0.84 (F1 0.816)** | 0.87 (0.856) | 0.87 (0.856) |
| GunPoint | 1.00 (1.000) | 1.00 (1.000) | 1.00 (1.000) |

Runtime: 0.21–0.35 s for every variant, with no meaningful difference.

**Conclusions:**

- **Hard counting on the max channel hurts** (−3 pp on ECG200). This supports the authors' choice of soft counting for the max channel.
- **Changing the min channel from hard to soft made no difference.** `soft_only` and the original are identical on all three datasets, with the same F1.
- **The earlier "improvement" was an artefact of the bugs.** Before the fixes, `soft_only` appeared to beat the original (0.90 vs 0.89 on ECG200). That result came from the buggy class and the default alphas, and **does not hold**.
- **Consistency check:** the original Hydra here gives 0.87 on ECG200, which exactly matches Notebook 1 (seed 0). The pipelines are consistent.

### 2.3 Experiment 3: k/g sensitivity (Coffee, ItalyPowerDemand, Wafer; seeds 0–2)

| Dataset | Config | Mean acc | Std | Mean runtime (s) |
|---|---|--:|--:|--:|
| Coffee | k4_g128 / k8_g64 / k16_g32 | 1.0000 (all) | 0 | 0.287 / 0.280 / **0.241** |
| ItalyPowerDemand | k4_g128 | 0.9666 | 0.0031 | 0.301 |
| ItalyPowerDemand | k8_g64 | **0.9689** | 0.0029 | 0.287 |
| ItalyPowerDemand | k16_g32 | 0.9653 | 0.0024 | **0.282** |
| Wafer | k4_g128 / k8_g64 / k16_g32 | 1.0000 (all) | 0 | 12.89 / 12.01 / **11.54** |

**Conclusion:** the spread across configurations on ItalyPowerDemand (0.36 pp) is about the same as the seed-to-seed std (about 0.3 pp), so accuracy is insensitive to k/g. `k16_g32` was the fastest on all three datasets (about 4% faster than the default on Wafer). This agrees with the 10-seed study in Notebook 1, and is a duplicate of it on the same three near-ceiling datasets.

**The notebook's overall conclusion:** none of the tested modifications improved accuracy over the original Hydra. The original design choices hold up, the default k/g is fine, and max-only is a possible efficiency gain that is worth testing on harder datasets.

Files: `results/Mahak/ablation_max_min.csv`, `results/Mahak/ablation_counting_methods_v2.csv` and `results/Mahak/ablation_kg_sensitivity.csv`.

---

## 3. Notebook 3: `aarti_combined_experiments.ipynb` (Aarti): comparison with Rocket-family models (corrected and re-run)

**Status:** fixed and re-run on 2026-11-07 on the same Mac as Notebook 1. Results are in `results/Aarti/`. Three issues were fixed before the re-run:

1. **Hydra+MultiRocket scaling bug.**
   - Only the Hydra features were scaled (`SparseScaler`). The raw MultiRocket features were concatenated unscaled, with a mean absolute value about 20× larger (14.7 vs 0.76 on ECG200; the maximum raw MultiRocket value was 95).
   - Because ridge applies a single penalty to all features, the Hydra block was effectively ignored, and Hydra+MultiRocket produced **predictions identical to MultiRocket alone** on all 6 datasets.
   - Fix: apply `StandardScaler` to the MultiRocket block (fitted on training data only) before concatenation. This matches aeon's `MultiRocketHydraClassifier`, which uses SparseScaler for Hydra, StandardScaler for MultiRocket, then concatenation and `RidgeClassifierCV(logspace(-3, 3, 10))`.
2. **Mismatched summary numbers.** The markdown summary reported timings that did not match the computed outputs. It also claimed Hydra+MultiRocket trained faster than MultiRocket, which was false. The summary has been rewritten from the new outputs.
3. **Wafer benchmark mixing.** The benchmark appended to a CSV that held 15 runs from the buggy pipeline. It now writes a fresh file each session (5 runs).

### 3.1 Setup

- **Datasets:** Coffee, ECG200, GunPoint, ItalyPowerDemand, TwoLeadECG and Wafer. Default split, `random_state = 42` (Notebooks 1 and 2 use seed 0), Ridge alphas `logspace(-3, 3, 10)`.
- **MiniRocket** (aeon): 10,000 kernels, max 32 dilations per kernel, giving 9,996 features.
- **MultiRocket** (aeon): 6,250 kernels, max 32 dilations, 4 features per kernel, `normalise=False`, giving 49,728 features. Its features are fed raw to Ridge in the standalone model.
- **Hydra + MultiRocket:** official Hydra (k=8, g=64, seed 42) scaled with `SparseScaler`, plus MultiRocket scaled with `StandardScaler`, concatenated and fed to Ridge. This gives 51,776–55,872 features (Hydra adds 2,048–6,144).

### 3.2 Six-dataset comparison (single run each; corrected pipeline)

| Dataset | MiniRocket acc (F1) | MultiRocket acc (F1) | Hydra+MultiRocket acc (F1) | H+MR vs MR | Paper Hydra+MultiRocket (30 resamples) |
|---|--:|--:|--:|--:|--:|
| Coffee | 1.0000 (1.0000) | 1.0000 (1.0000) | 1.0000 (1.0000) | tie | 1.0000 |
| ECG200 | 0.8900 (0.8783) | 0.9100 (0.9017) | **0.9200 (0.9121)** | +1.00 pp | 0.8917 |
| GunPoint | 0.9933 (0.9933) | 0.9933 (0.9933) | **1.0000 (1.0000)** | +0.67 pp | 0.9987 |
| ItalyPowerDemand | 0.9650 (0.9650) | **0.9728 (0.9728)** | 0.9679 (0.9679) | −0.49 pp | 0.9640 |
| TwoLeadECG | 0.9947 (0.9947) | **0.9991 (0.9991)** | 0.9982 (0.9982) | −0.09 pp | 0.9979 |
| Wafer | 0.9985 (0.9962) | 0.9992 (0.9979) | **0.9994 (0.9983)** | +0.02 pp | 0.9998 |
| **Mean** | **0.9736 (0.9713)** | **0.9791 (0.9775)** | **0.9809 (0.9794)** | +0.18 pp | |

- The MiniRocket and MultiRocket accuracies are identical to the pre-fix run, as expected.
- Hydra+MultiRocket now differs from MultiRocket, which confirms the fix works.
- Hydra+MultiRocket vs MultiRocket: **3 wins, 1 tie, 2 losses.** The largest win is one test case on ECG200 (100 test cases).
- For reference, plain Hydra in Notebook 1 (seed 0) scored ECG200 0.8700, GunPoint 1.0000, Coffee 1.0000, ItalyPowerDemand 0.9660, TwoLeadECG 0.9991 and Wafer 1.0000, a **mean of 0.9725**. Plain Hydra is therefore slightly below MiniRocket on these six datasets, but has the best score on GunPoint and Wafer.

**Timing (seconds, train / end-to-end test, on the Mac):**

| Dataset | MiniRocket | MultiRocket | Hydra+MultiRocket |
|---|--:|--:|--:|
| Coffee | 9.82* / 0.02 | 0.26 / 0.17 | 0.41 / 0.30 |
| ECG200 | 0.06 / 0.03 | 0.33 / 0.26 | 0.46 / 0.37 |
| GunPoint | 0.05 / 0.06 | 0.18 / 0.41 | 0.31 / 0.68 |
| ItalyPowerDemand | 0.03 / 0.11 | 0.08 / 0.78 | 0.13 / 1.05 |
| TwoLeadECG | 0.03 / 0.24 | 0.08 / 2.34 | 0.13 / 3.56 |
| Wafer | 0.71 / 2.41 | 4.30 / 23.98 | 6.69 / 34.87 |
| **Mean** | 1.78* (≈0.18 without Coffee) / **0.48** | 0.87 / 4.66 | 1.35 / 6.81 |

MiniRocket's 9.82 s on Coffee, the first dataset, is most likely one-off numba JIT compilation, not real training cost.

Standalone Hydra in Notebook 1 took 1.82 s to train and 9.79 s to test on Wafer (seed 0, same machine). It is faster at training than MultiRocket (4.30 s), but slower at test time than MiniRocket (2.41 s).

### 3.3 Repeated Wafer benchmark (5 runs, corrected pipeline)

| Metric | MiniRocket | MultiRocket | Hydra+MultiRocket |
|---|--:|--:|--:|
| Mean accuracy (std) | 0.998540 (0) | 0.999189 (0) | **0.999351** (0) |
| Mean macro F1 | 0.996205 | 0.997886 | **0.998310** |
| Mean train s (std) | **0.68** (0.01) | 4.22 (0.04) | 6.31 (0.26) |
| Mean end-to-end test s (std) | **2.36** (0.01) | 22.58 (0.20) | 33.14 (0.19) |

**Hydra+MultiRocket breakdown (mean of 5 runs):**

| Stage | Hydra transform | MultiRocket transform | Classifier |
|---|--:|--:|--:|
| Training | 1.58 s | 3.86 s | 0.87 s (fit) |
| Test | 9.71 s | 23.20 s | 0.23 s (prediction) |

Accuracy std is 0 because **every repeat uses the same random_state**, so the repeats measure timing noise only, not variance across seeds. On Wafer, Hydra's improvement is one extra correct test case out of 6,164.

**The notebook's conclusion:** with the bug fixed, Hydra+MultiRocket has the best mean accuracy and F1. However, the gain over MultiRocket is small (+0.18 pp), inconsistent (3 wins, 1 tie, 2 losses), and costs about 1.5× more training and test time. MiniRocket remains the best accuracy-to-speed trade-off: it is within about 0.7 pp of the best model and 10–14× faster at test time.

### 3.4 Remaining limitations of Notebook 3

- One seed (42) and one split per model and dataset. There is no variance estimate and no significance test.
- Four of the six datasets are at or above 99.3% accuracy for every model, which leaves almost no room to tell the models apart.
- **The block scaling is unweighted.** Hydra contributes 2–6k features against about 50k MultiRocket features under one ridge penalty, so relative block weighting has not been explored.
- **The alpha grid may be too narrow.** In a quick check outside the notebook (ECG200 and ItalyPowerDemand), RidgeCV chose an alpha at the edge of the `logspace(-3, 3, 10)` grid in every fit: 0.001 for MultiRocket on ECG200, and 1000 for MultiRocket on ItalyPowerDemand and for the fixed Hydra+MultiRocket on both datasets. This has not been checked systematically.
- Aarti's data loading uses aeon's default download directory, not `data/ucr_tsml`. This is a minor inconsistency.

---

## 4. Cross-notebook synthesis

All three notebooks have now been run on the same machine (a Mac with Apple Silicon, CPU only, PyTorch 2.15, aeon 1.6), so **timings are comparable across notebooks**.

### 4.1 What has been established

1. **The baseline is reproduced on all 112 UCR datasets.**
   - Mean accuracy is 0.8577, against 0.8735 in the paper. The median difference is about 0, and 59 of 112 datasets are within ±1 pp.
   - The remaining gap is concentrated in about 12 datasets. It is plausibly due to protocol (one default split and one seed, against a 30-resample mean) and, for the EOG datasets, missing input z-normalisation.
2. **k/g (with k×g = 512) has negligible effect on accuracy.** This was confirmed independently twice, with 10 seeds and with 3 seeds. Larger k (`k16_g32`) is consistently slightly faster.
3. **Counting ablations: the original Hydra design is not beaten.**
   - Soft max with hard min (the original) ties with soft/soft and beats hard/hard by 3 pp on ECG200.
   - The earlier apparent gain for soft/soft was a bug artefact.
4. **Max-only Hydra** halves the features and is about 20% faster with no loss, but this was only shown on Coffee, which is at ceiling.
5. **Hydra+MultiRocket (after the scaling fix) is the most accurate model on the 6 datasets**, with a mean of 0.9809 against 0.9791 for MultiRocket, 0.9736 for MiniRocket and 0.9725 for Hydra alone. The gain over MultiRocket is small, inconsistent and costs about 1.5× compute.
6. **Bugs found and fixed so far:**
   - Mahak's class indentation.
   - Mahak's Ridge alphas.
   - Aarti's missing MultiRocket scaling, which had been nullifying Hydra.
   - Aarti's stale summary numbers.
   - Aarti's mixed benchmark runs.
   - **Lesson:** every claimed improvement so far that appeared before verification was a bug.

### 4.2 Known gaps and weaknesses (to address)

- **Evaluation protocol:** only the default split, with a single seed. There are no resamples, so the results are not like-for-like with the paper's 30 resamples, and there is no proper significance testing.
- **Datasets used for ablations and comparisons:**
  - Everything apart from the 112-dataset baseline uses 3–6 easy datasets, most of them at or near 100%.
  - Nothing has been tested on the hard datasets from §1.2: Phoneme, EOG*, InlineSkate, Haptics, EthanolLevel, SemgHandMovement, the Refrigeration/Computers/ScreenType family, and the *Phalanx TW family.
- **Seeds:** Exp 1 and Exp 2 in the ablations used 1 seed, and Aarti's comparison used 1 seed.
- **Planned ablations not done:** first-order differencing on/off and clipping on/off.
- **No statistical comparison** across many datasets: no critical-difference diagram, no mean rank, and no Wilcoxon test of any variant against the original.
- **k/g only explored along k×g = 512.** Still unexplored:
  - total kernel count
  - kernel length (fixed at 9)
  - dilation schedule
  - k=1 (ROCKET-like) and very large k (dictionary-like) extremes
- **Input normalisation** is not handled. This affects the EOG datasets, whose raw std is about 91.
- **Hydra+MultiRocket block weighting** and the Ridge alpha range have not been tuned.
- **Unused official tools:** `softmax.py` (SGD for large datasets such as ElectricDevices and Crop) and `hydra_multivariate.py`.
- **Mahak and Aarti's experiments have not been run on all 112 datasets.** Notebook 1's 112-dataset loop could be reused.

### 4.3 Candidate directions already suggested by the data

- Run the counting variants (`hard_only`, `soft_only`, `max_only`) and Hydra+MultiRocket on **all 112 datasets**, with multiple seeds or resamples. One full Hydra run on the Mac takes about 5 minutes for training and 9 minutes for testing.
- Add **per-series z-normalisation** as an option. Test it on the EOG datasets and across all 112.
- Focus improvement ideas on the weak regimes: **long series (>700)**, **many classes (>10)** and **n_train of 200–1000**.
- Explore Hydra+MultiRocket **block weighting** and a wider alpha grid, and compare against the paper's mean of 0.8839 over 112 datasets.
- Max-only Hydra as a **speed variant**: test whether the 2× feature reduction costs any accuracy on the 112 datasets.

---

## Appendix: per-dataset results for default Hydra on all 112 UCR datasets (k=8, g=64, seed 0, default split)

Columns:

- **"Paper Hydra acc"** and **"Paper Hydra+MultiRocket"**: the authors' 30-resample means.
- **"Diff"**: our accuracy minus Paper Hydra, in percentage points.
- **"Train s" and "Test s"**: our timings on a Mac with Apple Silicon (CPU).

| Dataset | n_train | n_test | length | classes | Our acc | Our macro-F1 | Paper Hydra acc | Diff (pp) | Paper Hydra+MultiRocket | Train s | Test s |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ACSF1 | 100 | 100 | 1460 | 10 | 0.8600 | 0.8603 | 0.8067 | +5.33 | 0.8327 | 2.07 | 1.87 |
| Adiac | 390 | 391 | 176 | 37 | 0.8184 | 0.8097 | 0.8158 | +0.26 | 0.8240 | 0.76 | 0.73 |
| ArrowHead | 36 | 175 | 251 | 3 | 0.8343 | 0.8343 | 0.8697 | -3.54 | 0.8971 | 0.12 | 0.48 |
| BME | 30 | 150 | 128 | 3 | 1.0000 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 0.06 | 0.18 |
| Beef | 30 | 30 | 470 | 5 | 0.8333 | 0.8352 | 0.8056 | +2.78 | 0.7822 | 0.22 | 0.20 |
| BeetleFly | 20 | 20 | 512 | 2 | 0.9000 | 0.8990 | 0.9517 | -5.17 | 0.9067 | 0.17 | 0.17 |
| BirdChicken | 20 | 20 | 512 | 2 | 0.9000 | 0.8990 | 0.9183 | -1.83 | 0.8967 | 0.19 | 0.16 |
| CBF | 30 | 900 | 128 | 3 | 0.9933 | 0.9933 | 0.9919 | +0.15 | 0.9933 | 0.06 | 1.14 |
| Car | 60 | 60 | 577 | 4 | 0.9333 | 0.9333 | 0.9278 | +0.56 | 0.9261 | 0.40 | 0.38 |
| Chinatown | 20 | 343 | 24 | 2 | 0.9825 | 0.9784 | 0.9682 | +1.43 | 0.9671 | 0.01 | 0.09 |
| ChlorineConcentration | 467 | 3840 | 166 | 3 | 0.7599 | 0.7253 | 0.7634 | -0.35 | 0.7835 | 0.91 | 7.58 |
| CinCECGTorso | 40 | 1380 | 1639 | 4 | 0.9949 | 0.9949 | 0.9941 | +0.08 | 0.9696 | 0.88 | 27.00 |
| Coffee | 28 | 28 | 286 | 2 | 1.0000 | 1.0000 | 0.9976 | +0.24 | 1.0000 | 0.15 | 0.14 |
| Computers | 250 | 250 | 720 | 2 | 0.7040 | 0.7038 | 0.8185 | -11.45 | 0.8624 | 2.07 | 1.86 |
| CricketX | 390 | 390 | 300 | 12 | 0.8000 | 0.7986 | 0.8066 | -0.66 | 0.8251 | 1.31 | 1.24 |
| CricketY | 390 | 390 | 300 | 12 | 0.8410 | 0.8426 | 0.8284 | +1.26 | 0.8517 | 1.35 | 1.26 |
| CricketZ | 390 | 390 | 300 | 12 | 0.8051 | 0.7906 | 0.8226 | -1.74 | 0.8432 | 1.32 | 1.26 |
| Crop | 7200 | 16800 | 46 | 24 | 0.7336 | 0.7261 | 0.7348 | -0.12 | 0.7769 | 15.18 | 8.61 |
| DiatomSizeReduction | 16 | 306 | 345 | 4 | 0.9477 | 0.9374 | 0.9401 | +0.76 | 0.9522 | 0.11 | 1.08 |
| DistalPhalanxOutlineAgeGroup | 400 | 139 | 80 | 3 | 0.7554 | 0.7579 | 0.8062 | -5.08 | 0.7974 | 0.40 | 0.13 |
| DistalPhalanxOutlineCorrect | 600 | 276 | 80 | 2 | 0.7826 | 0.7703 | 0.8310 | -4.84 | 0.8412 | 0.60 | 0.26 |
| DistalPhalanxTW | 400 | 139 | 80 | 6 | 0.7122 | 0.5458 | 0.7010 | +1.13 | 0.6969 | 0.43 | 0.12 |
| ECG200 | 100 | 100 | 96 | 2 | 0.8700 | 0.8562 | 0.8947 | -2.47 | 0.8917 | 0.11 | 0.10 |
| ECG5000 | 500 | 4500 | 140 | 5 | 0.9491 | 0.5970 | 0.9471 | +0.20 | 0.9473 | 0.81 | 6.88 |
| ECGFiveDays | 23 | 861 | 136 | 2 | 1.0000 | 1.0000 | 0.9986 | +0.14 | 0.9957 | 0.07 | 1.29 |
| EOGHorizontalSignal | 362 | 362 | 1250 | 12 | 0.5718 | 0.5822 | 0.8270 | -25.52 | 0.8716 | 5.33 | 5.16 |
| EOGVerticalSignal | 362 | 362 | 1250 | 12 | 0.4945 | 0.4732 | 0.7640 | -26.95 | 0.8144 | 5.22 | 5.13 |
| Earthquakes | 322 | 139 | 512 | 2 | 0.7410 | 0.4744 | 0.7432 | -0.22 | 0.7441 | 1.71 | 0.66 |
| ElectricDevices | 8926 | 7711 | 96 | 7 | 0.7531 | 0.6724 | 0.8817 | -12.86 | 0.9039 | 36.23 | 7.47 |
| EthanolLevel | 504 | 500 | 1751 | 4 | 0.5860 | 0.5695 | 0.6643 | -7.83 | 0.7240 | 9.41 | 8.99 |
| FaceAll | 560 | 1690 | 131 | 14 | 0.8988 | 0.9158 | 0.9843 | -8.55 | 0.9839 | 0.91 | 2.49 |
| FaceFour | 24 | 88 | 350 | 4 | 0.8636 | 0.8774 | 0.8883 | -2.46 | 0.9008 | 0.17 | 0.33 |
| FacesUCR | 200 | 2050 | 131 | 14 | 0.9561 | 0.9440 | 0.9618 | -0.57 | 0.9696 | 0.30 | 3.03 |
| FiftyWords | 450 | 455 | 270 | 50 | 0.8418 | 0.7301 | 0.8272 | +1.46 | 0.8459 | 1.32 | 1.21 |
| Fish | 175 | 175 | 463 | 7 | 0.9886 | 0.9889 | 0.9863 | +0.23 | 0.9842 | 0.74 | 0.70 |
| FordA | 3601 | 1320 | 500 | 2 | 0.9606 | 0.9606 | 0.9585 | +0.21 | 0.9562 | 25.92 | 6.40 |
| FordB | 3636 | 810 | 500 | 2 | 0.8247 | 0.8246 | 0.9322 | -10.75 | 0.9294 | 26.24 | 3.90 |
| FreezerRegularTrain | 150 | 2850 | 301 | 2 | 0.9975 | 0.9975 | 0.9926 | +0.50 | 0.9986 | 0.48 | 8.93 |
| FreezerSmallTrain | 28 | 2850 | 301 | 2 | 0.9267 | 0.9267 | 0.9519 | -2.53 | 0.9914 | 0.13 | 8.97 |
| GunPoint | 50 | 150 | 150 | 2 | 1.0000 | 1.0000 | 0.9984 | +0.16 | 0.9987 | 0.11 | 0.24 |
| GunPointAgeSpan | 135 | 316 | 150 | 2 | 1.0000 | 1.0000 | 0.9963 | +0.37 | 0.9962 | 0.27 | 0.51 |
| GunPointMaleVersusFemale | 135 | 316 | 150 | 2 | 1.0000 | 1.0000 | 0.9993 | +0.07 | 1.0000 | 0.22 | 0.51 |
| GunPointOldVersusYoung | 136 | 315 | 150 | 2 | 1.0000 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 0.26 | 0.51 |
| Ham | 109 | 105 | 431 | 2 | 0.7429 | 0.7425 | 0.8476 | -10.48 | 0.8502 | 0.46 | 0.43 |
| HandOutlines | 1000 | 370 | 2709 | 2 | 0.9378 | 0.9312 | 0.9455 | -0.77 | 0.9505 | 31.56 | 11.68 |
| Haptics | 155 | 308 | 1092 | 5 | 0.5130 | 0.5028 | 0.5164 | -0.35 | 0.5319 | 1.93 | 3.71 |
| Herring | 64 | 64 | 512 | 2 | 0.7500 | 0.7374 | 0.6182 | +13.18 | 0.6104 | 0.37 | 0.35 |
| HouseTwenty | 40 | 119 | 2000 | 2 | 0.9580 | 0.9568 | 0.9689 | -1.09 | 0.9773 | 1.04 | 2.55 |
| InlineSkate | 100 | 550 | 1882 | 7 | 0.4909 | 0.4954 | 0.4959 | -0.50 | 0.5081 | 2.12 | 11.55 |
| InsectEPGRegularTrain | 62 | 249 | 601 | 3 | 1.0000 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 0.43 | 1.65 |
| InsectEPGSmallTrain | 17 | 249 | 601 | 3 | 1.0000 | 1.0000 | 0.9976 | +0.24 | 0.9969 | 0.18 | 1.67 |
| InsectWingbeatSound | 220 | 1980 | 256 | 11 | 0.6545 | 0.6433 | 0.6600 | -0.54 | 0.6649 | 0.51 | 4.68 |
| ItalyPowerDemand | 67 | 1029 | 24 | 2 | 0.9660 | 0.9660 | 0.9616 | +0.44 | 0.9640 | 0.03 | 0.29 |
| LargeKitchenAppliances | 375 | 375 | 720 | 3 | 0.8533 | 0.8534 | 0.9356 | -8.22 | 0.9413 | 2.74 | 2.68 |
| Lightning2 | 60 | 61 | 637 | 2 | 0.7213 | 0.7164 | 0.7377 | -1.64 | 0.7574 | 0.45 | 0.44 |
| Lightning7 | 70 | 73 | 319 | 7 | 0.8219 | 0.8175 | 0.7831 | +3.88 | 0.8041 | 0.31 | 0.26 |
| Mallat | 55 | 2345 | 1024 | 8 | 0.9407 | 0.9399 | 0.9557 | -1.50 | 0.9545 | 0.60 | 24.67 |
| Meat | 60 | 60 | 448 | 3 | 0.9167 | 0.9161 | 0.9839 | -6.72 | 0.9856 | 0.29 | 0.28 |
| MedicalImages | 381 | 760 | 99 | 10 | 0.7776 | 0.7178 | 0.7687 | +0.89 | 0.8086 | 0.47 | 0.78 |
| MiddlePhalanxOutlineAgeGroup | 400 | 154 | 80 | 3 | 0.5974 | 0.4889 | 0.6632 | -6.58 | 0.6530 | 0.39 | 0.13 |
| MiddlePhalanxOutlineCorrect | 600 | 291 | 80 | 2 | 0.8282 | 0.8218 | 0.8343 | -0.61 | 0.8515 | 0.63 | 0.27 |
| MiddlePhalanxTW | 399 | 154 | 80 | 6 | 0.5325 | 0.3585 | 0.5578 | -2.53 | 0.5515 | 0.39 | 0.13 |
| MixedShapesRegularTrain | 500 | 2425 | 1024 | 5 | 0.9806 | 0.9803 | 0.9818 | -0.12 | 0.9805 | 5.37 | 26.14 |
| MixedShapesSmallTrain | 100 | 2425 | 1024 | 5 | 0.9645 | 0.9642 | 0.9588 | +0.58 | 0.9564 | 0.99 | 25.95 |
| MoteStrain | 20 | 1252 | 84 | 2 | 0.9281 | 0.9273 | 0.9353 | -0.72 | 0.9072 | 0.04 | 1.21 |
| NonInvasiveFetalECGThorax1 | 1800 | 1965 | 750 | 42 | 0.9430 | 0.9382 | 0.9438 | -0.08 | 0.9621 | 15.81 | 16.17 |
| NonInvasiveFetalECGThorax2 | 1800 | 1965 | 750 | 42 | 0.9583 | 0.9546 | 0.9576 | +0.07 | 0.9681 | 15.78 | 15.73 |
| OSULeaf | 200 | 242 | 427 | 6 | 0.9876 | 0.9849 | 0.9788 | +0.88 | 0.9690 | 0.85 | 0.94 |
| OliveOil | 30 | 30 | 570 | 4 | 0.9333 | 0.9303 | 0.9122 | +2.11 | 0.9122 | 0.25 | 0.24 |
| PhalangesOutlinesCorrect | 1800 | 858 | 80 | 2 | 0.8298 | 0.8111 | 0.8338 | -0.39 | 0.8555 | 2.61 | 0.74 |
| Phoneme | 214 | 1896 | 1024 | 39 | 0.3238 | 0.1539 | 0.3297 | -0.59 | 0.3674 | 2.43 | 21.50 |
| PigAirwayPressure | 104 | 208 | 2000 | 52 | 0.7692 | 0.7589 | 0.7404 | +2.88 | 0.7234 | 2.28 | 4.48 |
| PigArtPressure | 104 | 208 | 2000 | 52 | 0.9856 | 0.9849 | 0.9516 | +3.40 | 0.9354 | 2.22 | 4.14 |
| PigCVP | 104 | 208 | 2000 | 52 | 0.9663 | 0.9661 | 0.9111 | +5.53 | 0.8535 | 2.26 | 4.35 |
| Plane | 105 | 105 | 144 | 7 | 1.0000 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 0.17 | 0.16 |
| PowerCons | 180 | 180 | 144 | 2 | 0.9778 | 0.9778 | 0.9620 | +1.57 | 0.9744 | 0.29 | 0.27 |
| ProximalPhalanxOutlineAgeGroup | 400 | 205 | 80 | 3 | 0.8634 | 0.7864 | 0.8454 | +1.80 | 0.8462 | 0.39 | 0.17 |
| ProximalPhalanxOutlineCorrect | 600 | 291 | 80 | 2 | 0.9141 | 0.8935 | 0.8970 | +1.71 | 0.9081 | 0.60 | 0.26 |
| ProximalPhalanxTW | 400 | 205 | 80 | 6 | 0.8146 | 0.4886 | 0.8059 | +0.88 | 0.8006 | 0.43 | 0.17 |
| RefrigerationDevices | 375 | 375 | 720 | 3 | 0.5067 | 0.5025 | 0.7561 | -24.94 | 0.7485 | 2.90 | 2.84 |
| Rock | 20 | 50 | 2844 | 4 | 0.9200 | 0.9194 | 0.8920 | +2.80 | 0.8640 | 0.98 | 1.93 |
| ScreenType | 375 | 375 | 720 | 3 | 0.4907 | 0.4915 | 0.6241 | -13.34 | 0.6779 | 2.81 | 2.73 |
| SemgHandGenderCh2 | 300 | 600 | 1500 | 2 | 0.8467 | 0.8359 | 0.8240 | +2.27 | 0.9446 | 5.47 | 10.36 |
| SemgHandMovementCh2 | 450 | 450 | 1500 | 6 | 0.4956 | 0.4971 | 0.4927 | +0.28 | 0.7717 | 7.72 | 7.71 |
| SemgHandSubjectCh2 | 450 | 450 | 1500 | 5 | 0.7711 | 0.7677 | 0.7806 | -0.95 | 0.9173 | 7.66 | 7.59 |
| ShapeletSim | 20 | 180 | 500 | 2 | 0.9889 | 0.9889 | 0.9837 | +0.52 | 1.0000 | 0.17 | 0.87 |
| ShapesAll | 600 | 600 | 512 | 60 | 0.9283 | 0.9270 | 0.9387 | -1.03 | 0.9483 | 3.15 | 2.98 |
| SmallKitchenAppliances | 375 | 375 | 720 | 3 | 0.8133 | 0.8123 | 0.8332 | -1.98 | 0.8366 | 2.81 | 2.69 |
| SmoothSubspace | 150 | 150 | 15 | 3 | 0.9267 | 0.9265 | 0.9693 | -4.27 | 0.9802 | 0.02 | 0.02 |
| SonyAIBORobotSurface1 | 20 | 601 | 70 | 2 | 0.9201 | 0.9198 | 0.9548 | -3.47 | 0.9487 | 0.03 | 0.53 |
| SonyAIBORobotSurface2 | 27 | 953 | 65 | 2 | 0.9475 | 0.9448 | 0.9509 | -0.34 | 0.9505 | 0.04 | 0.73 |
| StarLightCurves | 1000 | 8236 | 1024 | 3 | 0.9822 | 0.9730 | 0.9804 | +0.18 | 0.9815 | 10.63 | 84.98 |
| Strawberry | 613 | 370 | 235 | 2 | 0.9730 | 0.9708 | 0.9794 | -0.64 | 0.9797 | 1.51 | 0.85 |
| SwedishLeaf | 500 | 625 | 128 | 15 | 0.9776 | 0.9782 | 0.9706 | +0.70 | 0.9745 | 0.64 | 0.79 |
| Symbols | 25 | 995 | 398 | 6 | 0.9799 | 0.9802 | 0.9730 | +0.69 | 0.9724 | 0.15 | 3.77 |
| SyntheticControl | 300 | 300 | 60 | 6 | 0.9567 | 0.9573 | 0.9693 | -1.27 | 0.9947 | 0.25 | 0.21 |
| ToeSegmentation1 | 40 | 228 | 277 | 2 | 0.9605 | 0.9605 | 0.9430 | +1.75 | 0.9294 | 0.16 | 0.64 |
| ToeSegmentation2 | 36 | 130 | 343 | 2 | 0.9077 | 0.8558 | 0.9510 | -4.33 | 0.9438 | 0.20 | 0.49 |
| Trace | 100 | 100 | 275 | 4 | 1.0000 | 1.0000 | 1.0000 | +0.00 | 1.0000 | 0.40 | 0.35 |
| TwoLeadECG | 23 | 1139 | 82 | 2 | 0.9991 | 0.9991 | 0.9994 | -0.03 | 0.9979 | 0.05 | 1.18 |
| TwoPatterns | 1000 | 4000 | 128 | 4 | 0.9852 | 0.9853 | 0.9868 | -0.15 | 1.0000 | 1.62 | 5.20 |
| UMD | 36 | 144 | 150 | 3 | 0.9931 | 0.9931 | 0.9873 | +0.58 | 0.9850 | 0.09 | 0.27 |
| UWaveGestureLibraryAll | 896 | 3582 | 945 | 8 | 0.9726 | 0.9724 | 0.9737 | -0.11 | 0.9847 | 9.10 | 35.53 |
| UWaveGestureLibraryX | 896 | 3582 | 315 | 8 | 0.8529 | 0.8482 | 0.8635 | -1.06 | 0.8780 | 3.01 | 11.15 |
| UWaveGestureLibraryY | 896 | 3582 | 315 | 8 | 0.7926 | 0.7908 | 0.7931 | -0.05 | 0.8131 | 3.04 | 11.22 |
| UWaveGestureLibraryZ | 896 | 3582 | 315 | 8 | 0.8068 | 0.8041 | 0.7980 | +0.88 | 0.8216 | 3.06 | 11.20 |
| Wafer | 1000 | 6164 | 152 | 2 | 1.0000 | 1.0000 | 1.0000 | +0.00 | 0.9998 | 1.84 | 9.89 |
| Wine | 57 | 54 | 234 | 2 | 0.9074 | 0.9074 | 0.9512 | -4.38 | 0.9302 | 0.17 | 0.15 |
| WordSynonyms | 267 | 638 | 270 | 25 | 0.7429 | 0.6115 | 0.7499 | -0.70 | 0.7889 | 0.79 | 1.73 |
| Worms | 181 | 77 | 900 | 5 | 0.7532 | 0.7217 | 0.7532 | -0.00 | 0.7420 | 1.77 | 0.69 |
| WormsTwoClass | 181 | 77 | 900 | 2 | 0.7792 | 0.7754 | 0.8052 | -2.60 | 0.7965 | 1.77 | 0.71 |
| Yoga | 300 | 3000 | 426 | 2 | 0.9263 | 0.9259 | 0.9256 | +0.07 | 0.9373 | 1.32 | 12.48 |
