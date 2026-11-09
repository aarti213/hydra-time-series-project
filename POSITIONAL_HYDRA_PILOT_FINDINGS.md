# Positional Hydra Pilot: Implementation, Verification and Findings

**Project:** COMP41850, improving the HYDRA time series classifier (team: Shreed, Mahak, Aarti)
**Notebook:** `notebooks/shreed_positional_hydra.ipynb` (run end to end on 2026-11-07, 0 errors)
**Outputs:**
- `results/Shreed/positional_hydra_pilot.csv`: 275 rows = 11 datasets × 5 splits × 5 variants
- `results/Shreed/positional_hydra_pilot_multirocket.csv`: 165 rows, from the optional MultiRocket check (it was switched on for this run)
- `results/Shreed/figures/positional_hydra_scatter.png`
- `results/Shreed/figures/positional_hydra_gains.png`
- `results/Shreed/figures/positional_hydra_cost.png`

**Runtime:** about 11 min for the main pilot (55 dataset/split runs, 0 failures) and about 12.5 min for the MultiRocket check.

---

## TL;DR

1. **The implementation is correct.** All 6 in-notebook tests passed. A second implementation, written independently, reproduces the notebook's features and the exact accuracy and alpha for 12 checked runs. A label-shuffle check shows no leakage.
2. **Overall the gain is small and inconsistent:**
   - `pyr124_same_kernels` (7× features): **+0.43 pp** mean, wins 7 / draws 0 / losses 4 across datasets.
   - `meanpos` (1.5× features): **+0.19 pp**, 7/1/3.
   - Both equal-budget pyramids **lose** (−1.00 pp and −0.68 pp).
   - Remove EthanolLevel and `pyr124_same_kernels` falls to **−0.23 pp**, and `meanpos` to −0.01 pp.
3. **Location helps where position carries meaning:**
   - Spectra: EthanolLevel **+7.0 pp** (verified genuine), Ham +1.9, Wine +0.7.
   - Location-by-design synthetic data: SmoothSubspace +2.5, SyntheticControl +1.3, TwoPatterns +0.7. `meanpos` on TwoPatterns cuts the error from 1.36% to 0.30%.
4. **Location hurts where position is noise:**
   - **ShapeletSim −5.4 pp** (the location-free control behaved exactly as the hypothesis predicted, but far worse than the 0.5 pp tolerance).
   - **Lightning7 −3.0 pp** (its events are not aligned in time).
5. **At equal feature budget, the extra groups win:** with g=10 or 22 instead of 64, the pyramid loses on 7 of 11 datasets. The location bins don't make up for the groups they replace. EthanolLevel is the exception (+3.4 pp at 1.09× features).
6. **Inside MultiRocket+Hydra the gain mostly disappears:** +0.04 pp mean (2/4/5). MultiRocket's own location pooling already covers it. Only EthanolLevel keeps a real gain (+1.5 pp).
7. **The ridge alpha sits at a grid edge in 96% of `pyr124_same_kernels` fits** (73% for plain Hydra). The team's `logspace(-3, 3, 10)` grid is again too narrow, which could be under- or over-stating the 7× variants.
8. **Verdict: NO-GO for a generic replacement of Hydra.** All four criteria fail or pass only partly. Worth one more targeted step: make location a **train-CV-selected option**, widen the alpha grid, and test on the 40 development datasets, focusing on spectra and aligned data. Otherwise stop.

---

## 1. Motivation and idea

- **What Hydra measures.** For every kernel, Hydra counts how often it wins the competition within its group, **summed over the whole series**. It is a *bag of patterns*: it knows **which** patterns occur, but not **where**.
- **What its competitors do.**
  - Dictionary methods (S-BOSS, TDE) gained accuracy by counting words separately in sub-windows (a *spatial pyramid*).
  - MultiRocket encodes location through MIPV pooling (mean index of positive values).
- **Hypothesis.** Coarse location should help where position matters: aligned ECG beats, image outlines, spectra (position = wavelength), and synthetic datasets built on order or location. It should **not** help on location-free problems such as ShapeletSim.

**Variant A, spatial pyramid.**
- For each level `S` in `levels` (e.g. `(1, 2, 4)`), the series is split into `S` equal, contiguous segments, and wins are counted separately in each one.
- Segment ids are `seg(t) = floor(t·S/T)`.
- The soft count adds the max value into bin `argmax_kernel·S + seg(t)`.
- The hard count adds 1 into bin `argmin_kernel·S + seg(t)`.
- Level 1 is exactly the official Hydra.

**Variant B, mean position.**
- Each time step gets a position `pos(t) = (t+1)/T`, in (0, 1].
- For each kernel, the feature is the mean `pos` of its max-wins, or **exactly 0 if it never wins**.

---

## 2. How the code was written

The notebook holds everything; no other repo file was modified. The official Hydra is imported **by file path** with `importlib`, to avoid being shadowed by the pip `hydra-core` package. The authors' code is not modified.

### 2.1 Facts read from the official `hydra.py` (confirmed by the tests)
- `Hydra(input_length, k=8, g=64, seed)`, with `divisor = min(2, g)` and `h = g // divisor`.
- `W` has shape `(num_dilations, divisor, k·h, 1, 9)`, drawn from N(0,1), mean-centred and L1-normalised.
- `dilations = 2**arange(floor(log2((L−1)/8)) + 1)` and `padding = 4·d`.
- `forward`:
  - For each dilation, and for raw X then `torch.diff(X)`: `conv1d` is reshaped to `(n, h, k, T)`, with **T = L for raw and L − 1 for diff**.
  - `count_max` = `scatter_add` of max values at the argmax over k (soft).
  - `count_min` = `scatter_add` of ones at the argmin (hard).
  - Order per block: `[count_max, count_min]`, then concatenated and flattened.

### 2.2 `PositionalHydra(input_length, k, g, seed, levels, mean_position)`
- **Kernels.** It builds an **official `Hydra`** with the same arguments and copies `W`, `dilations` and `paddings`, so the kernels are identical to plain Hydra for the same seed and g.
- **Forward pass.**
  - The conv and diff code is copied line by line from the official forward.
  - Per block it computes `max_values, max_indices = Z.max(2)` and `min_values, min_indices = Z.min(2)`. These are the official calls, not `argmin`, so tie-breaking on flat stretches is identical. On flat stretches every mean-centred kernel outputs exactly 0. I caught and fixed this before the run.
  - For each level it runs the two `scatter_add` calls into `(n, h, k·S)` tensors, with bins in kernel-major order.
  - Optionally it adds mean position: `pos_sum / wins.clamp(min=1)`.
- **Feature order per block:** `[S=1 soft, S=1 hard, S=2 soft, S=2 hard, S=4 soft, S=4 hard, (mean position)]`, with blocks in the official `(d, diff_index)` order.
- **Safety and API.** It asserts that every segment is non-empty (`T ≥ max(levels)`), even for the shortest series, SmoothSubspace (L=15, diff T=14). It provides `batch()` (same as official) and `n_features`.

### 2.3 Variants (all k = 8)
| Variant | levels | mean_position | g | Features vs Hydra | Purpose |
|---|---|---|---|---|---|
| `hydra` | (1,) | no | 64 | 1× | The authors' class, unchanged |
| `pyr124_same_kernels` | (1,2,4) | no | 64 | **7×** | Does location add information, with the same kernels? |
| `pyr124_equal_budget` | (1,2,4) | no | 10 | 1.09× | **The hard test:** location bins replace groups |
| `pyr12_equal_budget` | (1,2) | no | 22 | 1.03× | Milder pyramid at equal budget |
| `meanpos` | (1,) | yes | 64 | 1.5× | One mean-position feature per kernel |

Example feature counts (ECG200): hydra 4,096; pyr124_same_kernels 28,672; pyr124_equal_budget 4,480; pyr12_equal_budget 4,224; meanpos 6,144. The largest is EthanolLevel: 8,192 for hydra and 57,344 for pyr124_same_kernels.

### 2.4 Pipeline and protocol
- **Classifier:** `SparseScaler` (the authors', fitted on train only), then `RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))`.
  - The scaler's zero-masking treats empty pyramid bins and "never wins" (0) as absent.
  - Its square root is monotone, so the ordering of positions is preserved.
- **Memory:** the test set is transformed, scaled and predicted in chunks of 1,024 series. This bounds memory for 7× features on TwoPatterns (4,000 test series).
- **Datasets:** 11, each tagged with a group and a type.
  - Location-by-design (all Simulated): SmoothSubspace, SyntheticControl, TwoPatterns, CBF.
  - Control (Simulated): ShapeletSim.
  - Aligned real data: ECG200 (ECG), DistalPhalanxOutlineCorrect (Image), Lightning7 (Sensor), EthanolLevel, Ham and Wine (Spectro).
- **Splits:** resample 0 is the default UCR split. Resamples 1–4 are stratified: pool train and test, then draw the **same per-class train counts** with `RandomState(resample_id)`. The code asserts there is no overlap and that sizes and class counts match. The resample id is also the Hydra seed.
- **No test-set choices:** alpha comes from LOO-CV inside `RidgeClassifierCV` on train, and variants and levels are fixed in advance.
- **Optional MultiRocket check:**
  1. aeon `MultiRocket(n_kernels=6250, max_dilations_per_kernel=32, n_features_per_kernel=4)`, with `StandardScaler`.
  2. Hydra features with `SparseScaler`.
  3. Concatenate, then `RidgeClassifierCV`. This is the corrected pipeline from Aarti's notebook.

---

## 3. Verification

### 3.1 In-notebook tests (all PASS)
| Test | Result |
|---|---|
| 1. `PositionalHydra(levels=(1,))` == official Hydra (ECG200 L=96 and Beef L=470, seeds 0 and 1, train and test) | **bit-identical** (`torch.equal`); kernels identical |
| 2. Pyramid consistency: S=2 and S=4 bins summed over segments == level 1, in every block | Hard counts **exact**. Soft counts max abs diff 7.6e-6 (ECG200) and 5.3e-5 (Beef), i.e. float summation order. The level-1 part == official, exactly |
| 3. Segment ids by hand (L=20) | T=20: S=2 → [10, 10], S=4 → [5, 5, 5, 5]. T=19 (diff): [10, 9] and [5, 5, 5, 4]. Contiguous and never empty. Per-segment win totals equal the number of time steps in each segment |
| 4. Mean position | Hand-built kernel that only wins in the first half: **0.2625** (expected 0.2625, ≤ 0.5). An independent recount on ECG200 matches. 884 of 40,960 kernel slots never win and are **exactly 0**. All values lie in (0, 1] |
| 5. Feature counts | `2·k·h·Σlevels (+k·h)` per block, correct for all variants at L=96 and L=15 |
| 6. Baseline | ECG200 seed 0 = **0.8700**, matching Notebook 1 (alpha 215.4) |
| Extra | Resampling sizes, class counts and no-overlap checks pass on every pilot dataset; the segment-length assert passes on every pilot dataset |

### 3.2 Independent post-run verification (my own separate script)
| Check | Result |
|---|---|
| Notebook code == generated code | Only one change, `RUN_MULTIROCKET = True`, made by re-running the config cell after the pilot. The pilot itself ran on unchanged code |
| CSV structure | 275 rows = 11 × 5 × 5; `n_features` matches the formula in every row; n_train and n_test are identical across resamples |
| **Independent implementation.** It cuts the series into time slices instead of using the `kernel·S + seg` index arithmetic, and has its own split code and its own mean position (one-hot) | Features `allclose` to the notebook's for every checked variant |
| **Exact accuracy and alpha reproduced** for 12 runs: SmoothSubspace r3 (all 5 variants), ShapeletSim r0 (hydra, pyr124, meanpos), Ham r1 (hydra, pyr124), EthanolLevel r0 (hydra 0.5860, pyr124 0.6500) | **12/12 match the CSV exactly** |
| **Leakage check** on the flagged EthanolLevel gain (the notebook warned "gain > 5 pp"): refit pyr124 with **shuffled train labels** | Accuracy falls to **0.27** (chance ≈ 0.25). The +6.4 to +9 pp gain on EthanolLevel is real signal, not leakage |

---

## 4. Results

### 4.1 Mean ± std accuracy over the 5 splits, and gain over `hydra` (pp)
| Dataset | Group / Type | L | hydra | pyr124_same | pyr124_equal | pyr12_equal | meanpos | Δ pyr124_same | Δ pyr124_eq | Δ pyr12_eq | Δ meanpos |
|---|---|---|---|---|---|---|---|---|---|---|---|
| SmoothSubspace | loc / Sim | 15 | 0.9573 ± 0.024 | 0.9827 ± 0.011 | 0.9680 ± 0.011 | 0.9613 ± 0.011 | 0.9693 ± 0.008 | **+2.53** | +1.07 | +0.40 | +1.20 |
| SyntheticControl | loc / Sim | 60 | 0.9647 ± 0.009 | 0.9780 ± 0.006 | 0.9673 ± 0.008 | 0.9687 ± 0.011 | 0.9673 ± 0.005 | **+1.33** | +0.27 | +0.40 | +0.27 |
| TwoPatterns | loc / Sim | 128 | 0.9864 ± 0.001 | 0.9932 ± 0.001 | 0.9802 ± 0.002 | 0.9913 ± 0.003 | **0.9970 ± 0.001** | +0.69 | −0.62 | +0.50 | **+1.07** |
| CBF | loc / Sim | 128 | 0.9869 ± 0.012 | 0.9804 ± 0.011 | 0.9802 ± 0.006 | 0.9847 ± 0.009 | 0.9869 ± 0.010 | −0.64 | −0.67 | −0.22 | 0.00 |
| ShapeletSim | control / Sim | 500 | 0.9767 ± 0.014 | 0.9222 ± 0.019 | 0.9022 ± 0.029 | 0.9444 ± 0.019 | 0.9578 ± 0.022 | **−5.44** | **−7.44** | −3.22 | −1.89 |
| ECG200 | real / ECG | 96 | 0.8900 ± 0.021 | 0.8920 ± 0.019 | 0.8840 ± 0.035 | 0.8860 ± 0.023 | 0.8880 ± 0.022 | +0.20 | −0.60 | −0.40 | −0.20 |
| DistalPhalanxOC | real / Image | 80 | 0.8261 ± 0.032 | 0.8203 ± 0.027 | 0.8080 ± 0.039 | 0.8203 ± 0.028 | 0.8297 ± 0.027 | −0.58 | −1.81 | −0.58 | +0.36 |
| Lightning7 | real / Sensor | 319 | 0.8219 ± 0.041 | 0.7918 ± 0.058 | 0.7699 ± 0.047 | 0.7945 ± 0.027 | 0.7973 ± 0.039 | **−3.01** | **−5.21** | −2.74 | −2.47 |
| EthanolLevel | real / Spectro | 1751 | 0.6476 ± 0.037 | **0.7176 ± 0.040** | 0.6816 ± 0.041 | 0.6368 ± 0.041 | 0.6696 ± 0.039 | **+7.00** | **+3.40** | −1.08 | +2.20 |
| Ham | real / Spectro | 431 | 0.8152 ± 0.055 | 0.8343 ± 0.057 | 0.8248 ± 0.057 | 0.8210 ± 0.049 | 0.8267 ± 0.059 | +1.90 | +0.95 | +0.57 | +1.14 |
| Wine | real / Spectro | 234 | 0.9630 ± 0.035 | 0.9704 ± 0.031 | 0.9593 ± 0.042 | 0.9519 ± 0.059 | 0.9667 ± 0.036 | +0.74 | −0.37 | −1.11 | +0.37 |
| **Mean** | | | **0.8942** | **0.8984** | 0.8841 | 0.8873 | 0.8960 | **+0.43** | −1.00 | −0.68 | +0.19 |

### 4.2 Win/draw/loss vs `hydra` (descriptive only; 11 datasets × 5 splits is a pilot and the Wilcoxon test is underpowered; **no significance claims**)
| Variant | Datasets W/D/L | Splits W/D/L | Mean gain (pp) | Median gain (pp) | Wilcoxon p (reference only) | Mean gain without EthanolLevel |
|---|---|---|---|---|---|---|
| pyr124_same_kernels | 7/0/4 | 29/8/18 | +0.43 | +0.69 | 0.465 | **−0.23** |
| pyr124_equal_budget | 4/0/7 | 16/10/29 | −1.00 | −0.60 | 0.413 | −1.44 |
| pyr12_equal_budget | 4/0/7 | 18/6/31 | −0.68 | −0.40 | 0.153 | −0.64 |
| meanpos | 7/1/3 | 25/11/19 | +0.19 | +0.36 | 0.413 | −0.01 |

### 4.3 Gain by group and by type (pp)
| | pyr124_same | pyr124_equal | pyr12_equal | meanpos |
|---|---|---|---|---|
| **location_by_design** (4) | **+0.98** | +0.01 | +0.27 | +0.63 |
| **control** (ShapeletSim) | **−5.44** | −7.44 | −3.22 | −1.89 |
| **aligned_real** (6) | +1.04 | −0.61 | −0.89 | +0.23 |
| ECG (ECG200) | +0.20 | −0.60 | −0.40 | −0.20 |
| Image (DistalPhalanx) | −0.58 | −1.81 | −0.58 | +0.36 |
| Sensor (Lightning7) | −3.01 | −5.21 | −2.74 | −2.47 |
| **Spectro** (3) | **+3.22** | +1.33 | −0.54 | +1.24 |
| Simulated (5) | −0.31 | −1.48 | −0.43 | +0.13 |

### 4.4 MultiRocket check (mean over 5 splits)
| Dataset | MultiRocket | MR + hydra | MR + pyr124_same | Δ pyramid inside MR (pp) | Δ hydra over MR (pp) |
|---|---|---|---|---|---|
| SmoothSubspace | 0.9733 | 0.9733 | 0.9733 | 0.00 | 0.00 |
| SyntheticControl | 0.9940 | 0.9920 | 0.9913 | −0.07 | −0.20 |
| TwoPatterns | 1.0000 | 1.0000 | 1.0000 | 0.00 | 0.00 |
| CBF | 0.9904 | 0.9904 | 0.9882 | −0.22 | 0.00 |
| ShapeletSim | 1.0000 | 1.0000 | 1.0000 | 0.00 | 0.00 |
| ECG200 | 0.9180 | 0.9140 | 0.9080 | −0.60 | −0.40 |
| DistalPhalanxOC | 0.8377 | 0.8413 | 0.8406 | −0.07 | +0.36 |
| Lightning7 | 0.8329 | 0.8411 | 0.8384 | −0.27 | +0.82 |
| EthanolLevel | 0.6956 | 0.7024 | **0.7176** | **+1.52** | +0.68 |
| Ham | 0.8286 | 0.8286 | 0.8305 | +0.19 | 0.00 |
| Wine | 0.9667 | 0.9667 | 0.9667 | 0.00 | 0.00 |
| **Mean** | 0.9125 | 0.9136 | 0.9141 | **+0.04** (W/D/L 2/4/5) | +0.11 |

**MultiRocket is a much stronger base than Hydra on these datasets.** It scores 1.0 on ShapeletSim and TwoPatterns, against Hydra's 0.977 and 0.986. Its location pooling already captures what the pyramid adds, so the pyramid's standalone gains on the synthetic datasets disappear. EthanolLevel is the only dataset where the pyramid still adds something.

### 4.5 Ridge alpha
**Share of fits at a grid edge (0.001 or 1000):**

| Variant | Share at edge |
|---|---|
| hydra | 73% |
| pyr124_same_kernels | **96%** (43 fits at 0.001, 10 at 1000) |
| pyr124_equal_budget | 73% |
| pyr12_equal_budget | 73% |
| meanpos | 78% |
| MultiRocket runs | 91–95% |

**Median alpha per dataset (hydra / pyr124_same_kernels):**
- Bottom edge (0.001 / 0.001) on most datasets.
- Top edge (1000 / 1000) on DistalPhalanx and EthanolLevel.
- On TwoPatterns and ECG200 the choice moves from 215 to 0.001 when the pyramid is added.

**Interpretation.** The grid cannot express the regularisation these models want. The 7× variant almost always sits at an edge, so its accuracy may be limited by the grid itself, in either direction. This repeats the team's earlier alpha-grid finding (Detach-Hydra pilot, Mahak/Aarti notebooks).

### 4.6 Cost (median relative to `hydra`, over all runs)
| Variant | Features | Train transform | Test transform | Ridge fit |
|---|---|---|---|---|
| pyr124_same_kernels | 7.00× | 1.28× | 1.25× | **3.56×** |
| pyr124_equal_budget | 1.09× | 0.51× | 0.47× | 0.97× |
| pyr12_equal_budget | 1.03× | 0.64× | 0.60× | 0.98× |
| meanpos | 1.50× | 1.11× | 1.11× | 1.20× |

- **Absolute example, EthanolLevel (largest dataset):** test transform 9.1 s for hydra → 11.8 s for pyr124 → 9.9 s for meanpos. Ridge fit 0.09 s → 0.56 s.
- **The equal-budget variants are faster than hydra**, because g = 10 or 22 means fewer convolutions.
- **Memory is the bigger cost of the 7× variant:** 57,344 features per series on EthanolLevel.

---

## 5. Findings from the figures

### 5.1 `positional_hydra_scatter.png` (hydra accuracy on x vs variant accuracy on y; mean of 5 splits with ±std error bars; above the diagonal = gain)
- **Left, pyr124_same_kernels:**
  - **EthanolLevel stands far above the diagonal** (0.648 → 0.718). Its error bars barely overlap the diagonal, the only clear separation in the plot.
  - **ShapeletSim (red, the control) falls well below** (0.977 → 0.922). This is the other clear separation, in the opposite direction.
  - The location-by-design points (blue: SmoothSubspace, SyntheticControl, TwoPatterns) sit just above the diagonal in the top-right corner, near ceiling. CBF sits just below.
  - Lightning7 is below the diagonal and Ham above, but both have **very large error bars** (std 4–6 pp), which overlap the diagonal. ECG200 and DistalPhalanx lie on the diagonal.
- **Right, meanpos:**
  - Every point stays much closer to the diagonal. The variant is **lower risk**: smaller gains but much smaller losses (ShapeletSim only −1.9 pp).
  - TwoPatterns and EthanolLevel are above the diagonal; Lightning7 and ShapeletSim are slightly below.
- **Overall:** for most datasets the change is smaller than the split-to-split standard deviation. Only EthanolLevel (gain) and ShapeletSim (loss) clearly move.

### 5.2 `positional_hydra_gains.png` (per-dataset gain of each variant, in pp)
- **The two extremes are EthanolLevel (all bars except pyr12 positive, blue +7.0) and ShapeletSim (all bars negative, orange −7.4).** Lightning7 is the third clearly negative dataset, with every variant between −2.5 and −5.2.
- **Within a dataset the ranking is consistent:**
  - On the datasets that gain, the order is usually pyr124_same_kernels > meanpos > equal-budget variants.
  - On the datasets that lose, `pyr124_equal_budget` (orange) loses most. It has both location bins **and** only 10 groups, so it carries both costs.
- **`meanpos` (red) is the steadiest bar:** positive on 7 datasets and only mildly negative on the others, except Lightning7 and ShapeletSim.
- **`pyr12_equal_budget` (green) is small everywhere.** Its bars stay within ±0.6 pp on most datasets, so it neither helps nor hurts much, apart from ShapeletSim (−3.2), Lightning7 (−2.7) and Wine/EthanolLevel (about −1.1).

### 5.3 `positional_hydra_cost.png` (median cost relative to hydra)
- **`pyr124_same_kernels` is the expensive one:** 7× features and a **3.6× ridge fit**, but only a **1.25× test transform**. The convolutions are identical; only the extra scatter calls add time.
- **`meanpos` costs about 1.1–1.5× across the board.**
- **The equal-budget variants have 1.03–1.09× features, about 1× fit time and a 0.5–0.6× transform time**, so they are cheaper than Hydra. But they lose accuracy.

---

## 6. Interpretation

- **Location is dataset-specific information, not a free improvement.**
  - **Where position has a fixed meaning, it helps:** a wavelength in spectra (EthanolLevel, Ham, Wine), or a designed shape or location (SmoothSubspace, SyntheticControl, TwoPatterns).
  - **Where the discriminative pattern can occur anywhere, it hurts:** ShapeletSim (the shapelet is placed at random) and Lightning7 (unaligned events). There the bins split each kernel's evidence across segments, adding noise and dimensions without adding signal.
- **Why EthanolLevel gains so much.** Its classes (ethanol concentration) are identified by absorbance at particular wavelengths, i.e. positions. Plain Hydra pools over all 1,751 positions and is weak here (0.648). The pyramid restores the wavelength information.
  - The gain is verified: reproduced independently, chance-level with shuffled labels, and positive for three of the four variants, including the equal-budget pyramid (+3.4 pp at 1.09× features).
  - It is the one dataset where location beats more groups.
- **The equal-budget test fails.** Replacing groups with location bins (g=64 → g=10) loses 1.0 pp on average. This agrees with the paper's finding that more groups help. Location is worth less than kernel diversity on most datasets.
- **Inside MultiRocket the effect almost disappears** (+0.04 pp). The pyramid mostly re-encodes information MultiRocket already has. This sharply limits the practical value of Positional Hydra in a MultiRocket+Hydra ensemble.
- **CBF and ECG200 show no gain** despite "aligned" or "location" labels. CBF's shapes have random onset (location varies), and ECG200's beats are aligned, but its discriminative morphology appears to be captured well enough without location.

---

## 7. Go / no-go verdict

| Question | Answer | Evidence |
|---|---|---|
| Q1. Does `pyr124_same_kernels` or `meanpos` gain ≥ 1 pp on the four location-by-design datasets, without hurting ShapeletSim by more than 0.5 pp? | **NO** | pyr124: +0.98 pp (just short), but ShapeletSim −5.44. meanpos: +0.63 pp and ShapeletSim −1.89 |
| Q2. Does any variant gain on the aligned real datasets? | **Only on spectra** | Spectro +3.22 pp (pyr124) and +1.24 pp (meanpos), mostly from EthanolLevel. ECG +0.2 / −0.2, Image −0.6 / +0.4, Sensor −3.0 / −2.5 |
| Q3. Does the gain survive at equal feature budget? | **NO** | pyr124_equal −1.00 pp (4/0/7), pyr12_equal −0.68 pp (4/0/7). The only exceptions are EthanolLevel (+3.4) and SmoothSubspace (+1.1) |
| Q4. Is the extra cost acceptable? | **meanpos yes; pyr124 borderline** | meanpos about 1.1–1.2×. pyr124 has 7× features (memory) and a 3.6× fit, for +0.43 pp on average (−0.23 pp without EthanolLevel) |

**Recommendation: do not scale Positional Hydra up as a generic replacement for Hydra.**

- **Why:** the average gain is too small and depends on one dataset. The control is badly hurt. The equal-budget test fails. And the effect vanishes inside MultiRocket+Hydra.

**If the team wants to pursue it further, one targeted follow-up is justified:**
1. **Make location a selected option, not a default.** Choose `levels ∈ {(1,), (1,2,4)}`, or meanpos on/off, per dataset by **train-set CV**. The pilot shows a clear split between datasets that benefit (spectra, location by design) and datasets that are hurt (ShapeletSim, Lightning7), so a CV switch could keep the gains without the losses.
2. **Widen the alpha grid first**, e.g. `logspace(-3, 6, 20)` or larger. 96% of pyramid fits sit at an edge, which makes the current numbers uncertain.
3. **Test on the 40 development datasets with 5 resamples**, reporting results by type and especially for the spectro datasets (Beef, Coffee, Meat, OliveOil, Strawberry, EthanolLevel, Ham, Wine). Stop if the CV-selected version does not beat plain Hydra by ≥ 0.5 pp on average, or does not survive inside MultiRocket+Hydra.

---

## 8. Caveats

- **Pilot scale:** 11 datasets × 5 splits, so there are **no significance claims**. Standard deviations across splits are 1–6 pp on the real datasets, larger than most of the observed gains.
- **Resamples:** only 4 stratified resamples beyond the default split; the paper used 30.
- **Timings:** default thread count, one machine. They are only meaningful relative to each other.
- **Alpha grid:** the gains and losses of the 7× variant may partly reflect the grid, since 96% of its fits are at an edge.
- **Execution order:** the config cell was re-run after the pilot to set `RUN_MULTIROCKET = True`. This did not change the main pilot. I verified that the code is otherwise identical to the generated version.
- **Verdict cell:** the notebook's verdict markdown cell is still the template. This document is the filled-in verdict.
