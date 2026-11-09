# Detach-Hydra v2: confirmatory test on the 72 held-out UCR datasets — findings

**Notebook:** `notebooks/shreed_detach_hydra_confirm.ipynb` (full run, `QUICK_RUN = False`)
**Run date:** 2026-11-08
**Follows:** `shreed_detach_hydra_decisive.ipynb` / `DETACH_HYDRA_DECISIVE_FINDINGS.md` (development run on 40 datasets)
**Result files (`results/Shreed/`):**
- `detach_confirm_runs.csv`: 6,480 rows
- `detach_confirm_paths.csv`
- `detach_confirm_kept_groups.csv`
- `detach_confirm_devcheck_runs.csv`: Stage A, 660 rows
- figures in `results/Shreed/figures/detach_confirm_*.png`

---

## 1. Headline

**The pre-registered success rule is met at budget 0.0625 (1/16 of the kernel groups).** At that budget, on 72 datasets the method was not developed on, Detach v2 beats a fresh smaller Hydra (`small_g`) as follows:

| Measure | Value |
|---|---|
| Mean difference | **+1.22 pp** (95% CI +0.55 to +1.97) |
| Wins / draws / losses | **49 / 4 / 19** |
| Holm-corrected Wilcoxon p | **0.0005** |

At budget 0.125 it meets two of the three criteria: mean +0.56 pp and Holm p = 0.048. It misses the third, because 42 wins against 23 losses is 1.83×, short of the required 2×. At budgets 0.25 and 0.5 the difference is small and not significant.

So the rule as written gives **SUCCESS**. The effect is modest, though:
- **Pruning still costs accuracy.** Detach v2 is 1.9 pp below the full Hydra at 0.0625.
- **Training is slower.** Detach v2 takes about 3.2× as long to train as the full Hydra (median per split).
- **The test does not show that stability averaging (v2) caused the success.** v1 was not run on the held-out data (see §8).

---

## 2. What was tested (pre-registration, unchanged from the built notebook)

**Hypothesis.** At matched kernel budgets, Detach v2 is more accurate than a fresh smaller Hydra (`small_g`). Detach v2 ranks kernel groups by importance and prunes the weakest, with importance scores averaged over subsamples.

**Data**
- The 72 UCR datasets in the authors' 112-dataset list that were **not** among the 40 development datasets. The run checked that the two lists do not overlap.
- 5 splits each: the default UCR split plus stratified resamples 1–4.

**Methods**

| Method | Description |
|---|---|
| `full` | Official Hydra, k = 8, g = 64 |
| `detach_v2` | Train the full Hydra, then prune 10% of the groups per step. Each group's score is its ridge-coefficient L2 norm, normalised and averaged over **B = 5 fixed stratified 80% subsamples of the training set** (seeds 10,000 + 100r + b). |
| `small_g` | A fresh Hydra with g′ = 32 / 16 / 8 / 4, which matches each budget exactly; 3 kernel seeds, averaged |

- Every model uses `SparseScaler` (fitted on train only) and `RidgeClassifierCV(np.logspace(-6, 6, 25))`.
- `full_author_grid` (alphas `np.logspace(-3, 3, 10)`) is included for reference only.

**Budgets.** 0.5, 0.25, 0.125 and 0.0625 of the groups.

**Success rule.** At **any budget ≤ 0.25**, all three must hold:
1. mean Δ ≥ +0.5 pp;
2. wins ≥ 2 × losses, where a draw is |Δ| < 0.05 pp;
3. two-sided Wilcoxon p < 0.05 after Holm correction across the 4 budgets.

Δ is computed on per-dataset accuracy averaged over the 5 splits (and over the 3 seeds for `small_g`).

**Integrity of the run**
- The only edit made before running was `QUICK_RUN = True → False`. Two empty cells were also appended at the end. I diffed the notebook against the built version to confirm this.
- The pre-registration cell, Detach v2's settings and all the code are unchanged.
- No test data was used for any choice: ridge alpha comes from LOO-CV on training data, and the group ranking uses training data only. The run asserts the training-data-only signature.

---

## 3. Run summary and checks

**Run**
- **Completeness:** 72 of 72 held-out datasets × 5 splits = 360 of 360 pairs. Nothing failed, so no failures file was written.
- **Stage A:** 6 development datasets × 5 splits = 30 pairs, 8.9 min.
- **Stage B:** 437.7 min (7.3 h) in a single uninterrupted run, starting from "0 pairs already complete".
- **Load time:** all 78 datasets (72 held-out plus 6 for Stage A) loaded in 9 s.
- **Software:** numpy 2.3.5, pandas 3.0.6, torch 2.15.0, 10 threads.

**All 13 logged checks passed:**
- **Official Hydra structure:** divisor = 2, h = g/2, groups = num_dilations × g, and the dilations do not depend on g.
- **Training data only:** both pruning-path functions take training arrays only.
- **Dataset lists:** 112 = 40 development + 72 held-out, with no overlap.
- **Pilot Tests 1–4 on ECG200 and Beef:**
  - A `PrunedHydra` that keeps every group equals the official Hydra.
  - With random masks, the rebuilt model's features equal the kept columns of the full features.
  - Kernel and feature counts are correct.
  - Fitting the `SparseScaler` once and then taking the kept columns is exact (max |diff| = 0).
- **v2 generalises v1:** with B = 1 and frac = 1.0, v2 gives the same kept masks as v1 at all 4 budgets (ECG200, Beef).
- **Stratified subsampler:** every class is present, it is deterministic, and sizes are max(1, round(0.8·n_c)).
- **Budget sizes:** each v2 budget keeps exactly num_dilations × g′ groups, the same as `small_g`.
- **Rebuild equivalence:** the v2 mask at 0.25 equals the kept columns, on ECG200 before the run, and **inside Stage B on the first split of all 72 datasets**. Exact budget sizes were asserted on every pair.
- **Baseline sanity:** ECG200 seed 0 with the authors' grid gives 0.8700, matching Notebook 1.
- **Split checks:** the `make_split` asserts passed for 78 datasets × 5 resamples.

**Independent re-check.** I recomputed the primary table, the W/D/L counts, the Wilcoxon and the Holm p-values directly from `detach_confirm_runs.csv`, outside the notebook. They match the notebook to every printed digit.

---

## 4. Main results (72 held-out datasets, mean of per-dataset means)

The full Hydra scores **0.8844** with the wide alpha grid and 0.8855 with the authors' grid.

| Budget | Full | Detach v2 | small_g | v2 − full (pp) | small_g − full (pp) |
|---|---|---|---|---|---|
| 0.5 | 0.8844 | **0.8823** | 0.8810 | −0.21 | −0.34 |
| 0.25 | 0.8844 | **0.8768** | 0.8738 | −0.76 | −1.07 |
| 0.125 | 0.8844 | **0.8720** | 0.8663 | −1.25 | −1.81 |
| 0.0625 | 0.8844 | **0.8656** | 0.8534 | −1.88 | −3.10 |

Detach v2 has the higher mean at every budget, and its lead grows as the budget shrinks. At 1/16 of the groups, Detach v2 loses about 60% of what `small_g` loses relative to the full model (1.88 pp against 3.10 pp).

---

## 5. Primary test: `detach_v2 − small_g`

| Budget | Mean (pp) | 95% bootstrap CI | Median (pp) | W/D/L | p | Holm p | ≤ 0.25 | Mean ≥ 0.5 | W ≥ 2L | Holm < 0.05 | **All met** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | +0.13 | [−0.15, +0.44] | +0.08 | 38/8/26 | 0.2450 | 0.2911 | – | ✗ | ✗ | ✗ | ✗ |
| 0.25 | +0.31 | [−0.18, +0.86] | +0.14 | 39/8/25 | 0.1455 | 0.2911 | ✓ | ✗ | ✗ | ✗ | ✗ |
| 0.125 | +0.56 | [−0.03, +1.18] | +0.22 | 42/7/23 | 0.0161 | 0.0483 | ✓ | ✓ | ✗ (1.83×) | ✓ | ✗ |
| **0.0625** | **+1.22** | **[+0.55, +1.97]** | **+0.60** | **49/4/19** | **0.0001** | **0.0005** | ✓ | ✓ | ✓ (2.58×) | ✓ | **✓** |

**Rule outcome: met at 0.0625, so the pre-registered hypothesis is supported.**

**Observations**
- **The effect is concentrated at small budgets.** It is negligible at 0.5, grows steadily, and is clearly significant only at 0.0625. This matches the development run.
- **At 0.125 the shortfall is the win ratio.** Meeting it needed 46 wins against 23 losses, or 21 losses against 42 wins. The mean and the Holm p-value pass, only just.
- **The mean is larger than the median at every budget** (for example +1.22 against +0.60 at 0.0625), so a few datasets with large gains pull the mean up. Most datasets show small gains.
- **It holds on every split.** The mean Δ at 0.0625 is positive on each of the 5 splits taken separately: +1.32 / +0.96 / +1.19 / +1.36 / +1.29 pp for r0–r4. At 0.125 it is +0.36 to +0.83 on every split.
- **Macro-F1 points the same way and is stronger.** These are exploratory figures, not part of the rule:

  | Budget | Δ macro-F1 (pp) | Wins / losses | Unadjusted p |
  |---|---|---|---|
  | 0.5 | +0.35 | 41 / 22 | 0.023 |
  | 0.25 | +0.63 | 41 / 24 | 0.022 |
  | 0.125 | +1.00 | 43 / 22 | 0.0009 |
  | 0.0625 | +1.70 | 52 / 16 | < 0.0001 |

- **Robustness of the 0.0625 result** (exploratory, Wilcoxon p values unadjusted):

  | Change | Mean Δ (pp) | p |
  |---|---|---|
  | Drop PigAirwayPressure (the biggest gain, +17.5 pp) | +0.99 | 0.0002 |
  | Drop the top 3 (PigAirwayPressure, FaceFour, TwoPatterns) | +0.81 | 0.0005 |
  | Drop Earthquakes (the biggest loss) | +1.34 | < 0.0001 |

  No single dataset drives the result.

  At 0.125 the result is fragile. Without the top 3 gainers, the mean falls to +0.22 pp with p = 0.054.

**Largest per-dataset differences (pp)**

| Budget | Worst for Detach v2 | Best for Detach v2 |
|---|---|---|
| 0.125 | Earthquakes −8.35, DistalPhalanxOutlineAgeGroup −5.23, Computers −4.91, SmallKitchenAppliances −3.04, MiddlePhalanxTW −2.94 | PigAirwayPressure +13.85, FaceFour +6.29, TwoPatterns +5.54, ChlorineConcentration +4.07, Lightning2 +3.93 |
| 0.0625 | Earthquakes −7.05, Computers −3.89, DistalPhalanxOutlineAgeGroup −3.41, MiddlePhalanxTW −2.16, ShapesAll −1.72 | PigAirwayPressure +17.53, FaceFour +7.88, EthanolLevel +7.88, SemgHandSubjectCh2 +7.33, TwoPatterns +6.62 |

---

## 6. Secondary analyses (pre-declared, descriptive)

### 6.1 `detach_v2 − full`

| Budget | Mean (pp) | 95% CI | W/D/L | Holm p |
|---|---|---|---|---|
| 0.5 | −0.21 | [−0.49, +0.09] | 16/17/39 | 0.0045 |
| 0.25 | −0.76 | [−1.23, −0.28] | 13/10/49 | < 0.0001 |
| 0.125 | −1.25 | [−1.84, −0.65] | 12/9/51 | < 0.0001 |
| 0.0625 | −1.88 | [−2.55, −1.24] | 8/9/55 | < 0.0001 |

**Pruning is not free.** Even at half the groups, Detach v2 is significantly, if only slightly, worse than the full model: it loses on 39 datasets and wins on 16. The loss grows steadily as the budget shrinks.

### 6.2 Overfitting failure mode: datasets with `detach_v2 − small_g` < −5 pp at any budget

**2 of 72.**

| Dataset | Δ at 0.5 | Δ at 0.25 | Δ at 0.125 | Δ at 0.0625 | n_train | Classes | Length |
|---|---|---|---|---|---|---|---|
| Earthquakes | −3.88 | −4.99 | −8.35 | −7.05 | 322 | 2 | 512 |
| DistalPhalanxOutlineAgeGroup | −2.93 | −4.70 | −5.23 | −3.41 | 400 | 3 | 80 |

DistalPhalanxOutlineAgeGroup is the sister dataset of MiddlePhalanxOutlineAgeGroup, the outlier in the development run. Both failures are noisy problems with few classes and moderate n_train. This is the setting where ranking groups on training data can pick up noise.

### 6.3 Training cost

Wall-clock times use all cores. `path_s` is one pruning path, which serves all 4 budgets. The Detach v2 cost is full train transform + path + fit.

| | 0.5 | 0.25 | 0.125 | 0.0625 |
|---|---|---|---|---|
| Path, median per split (s) | 0.89 | 0.89 | 0.89 | 0.89 |
| Detach v2, median per split (s) | 2.19 | 2.18 | 2.17 | 2.17 |
| small_g, median per split (s) | 0.47 | 0.36 | 0.32 | 0.26 |
| Full, median per split (s) | 0.73 | 0.73 | 0.73 | 0.73 |
| v2 / small_g (median of pairs) | 4.6× | 5.9× | 8.0× | 9.4× |
| v2 / small_g (totals) | 14.6× | 22.2× | 32.1× | 40.7× |
| v2 / full (median of pairs) | 3.2× | 3.2× | 3.2× | 3.2× |
| v2 / full (totals) | 7.1× | 7.0× | 7.0× | 7.0× |

- **Pruning paths:** 2.2 h in total out of the 7.29 h of pair wall-clock.
- **Where the time went:** almost all of it on the large-n datasets. Path time summed over the 5 splits:

  | Dataset | Path time (5 splits) |
  |---|---|
  | ElectricDevices | 4,003 s, about 800 s per split |
  | Crop | 1,774 s |
  | NonInvasiveFetalECGThorax1 | 404 s |
  | NonInvasiveFetalECGThorax2 | 402 s |

  When n_train is larger than the number of features, each of the 5 subsample ridge fits per step is expensive.
- **Training cost is the price of v2.** It costs about 3× the full model and 5–9× a matched `small_g`, and much more on large datasets. The pruned model's **inference** cost equals `small_g`'s, since the kernel and feature counts are identical.
- **Inference proxy:** `small_g`'s test-transform speed-up over the full model has a median of 1.6 / 2.3 / 2.9 / 3.3× at the four budgets. These ratios are small because tiny test sets are dominated by fixed overhead. The decisive run timed `PrunedHydra` directly and got 1.9–10.2×.

### 6.4 Kept-group structure (Detach v2)

**Keep rate by relative dilation**, relative to uniform (1.0 = `small_g`'s even allocation):

| Budget | Short | Mid | Long |
|---|---|---|---|
| 0.0625 | 0.65 | 0.98 | **1.47** |
| 0.125 | 0.69 | 1.00 | 1.41 |
| 0.25 | 0.74 | 1.02 | 1.31 |
| 0.5 | 0.82 | 1.03 | 1.20 |

**Share of kept groups on diff(X)**, where 0.5 = uniform:

| Budget | Mean | Min | Max |
|---|---|---|---|
| 0.0625 | 0.347 | 0.000 | 0.860 |
| 0.125 | 0.386 | 0.000 | 0.860 |
| 0.25 | 0.413 | 0.004 | 0.835 |
| 0.5 | 0.455 | 0.156 | 0.799 |

**Block coverage**, the fraction of (dilation, input) blocks with at least one kept group. `small_g` is 1.0 at every budget by construction.

| Budget | Mean | Minimum over pairs |
|---|---|---|
| 0.0625 | 0.691 | 0.167 |
| 0.125 | 0.840 | 0.25 |
| 0.25 | 0.939 | 0.5 |
| 0.5 | 0.989 | 0.786 |

**Observations**
- **The same pattern as the development run.** Detach v2 moves budget to **long dilations** (1.47× uniform at 0.0625) and to the **raw series rather than diff(X)** (only 35% of kept groups on diff at 0.0625).
- **Some datasets keep no diff groups at all,** and at small budgets it gives up entire blocks.
- **This non-uniform allocation is what `small_g` cannot do,** and it is the likely source of the advantage.

### 6.5 Breakdowns of Δ (`detach_v2 − small_g`, pp)

| Group | n | Δ 0.5 | Δ 0.25 | Δ 0.125 | Δ 0.0625 |
|---|---|---|---|---|---|
| n_train ≤ 100 | 30 | +0.24 | +0.47 | +0.49 | +0.93 |
| n_train 101–500 | 27 | −0.07 | −0.11 | +0.23 | +1.22 |
| n_train > 500 | 15 | +0.30 | +0.73 | +1.30 | +1.81 |
| length ≤ 200 | 28 | +0.08 | +0.16 | +0.58 | +0.97 |
| length 201–500 | 12 | +0.54 | +1.09 | +1.22 | +2.03 |
| length > 500 | 32 | +0.02 | +0.14 | +0.30 | +1.14 |
| 2 classes | 28 | −0.20 | −0.13 | −0.09 | +0.33 |
| 3–10 classes | 32 | +0.18 | +0.15 | +0.67 | +1.50 |
| > 10 classes | 12 | +0.79 | +1.74 | +1.80 | **+2.54** |

**Observations**
- **Class count is the clearest pattern.** Detach v2 does not help on binary problems (about 0 down to 0.33 pp). Its gain grows with the number of classes, reaching +2.5 pp at 0.0625 for more than 10 classes.
- **Large training sets also benefit more:** +1.81 pp at 0.0625 when n_train > 500.
- **This fits the mechanism.** More classes and more training data give a more reliable estimate of group importance.

### 6.6 Ridge alpha at the grid edge

| Method | Fraction of fits at the edge | Which edge |
|---|---|---|
| Full | 21.9% | all at the low edge (1e-6) |
| Detach v2 | 29.4% | all at the low edge |
| small_g | 17.9% | all at the low edge |
| Full, authors' grid | 79.7% | 47% low, 33% high |

Inside the pruning path, the mean fraction of subsample fits at the edge was 0.31.

- **The wide grid fixes most edge hits.** The authors' grid hits an edge in 80% of fits.
- **The remaining hits are all at the low end,** meaning almost no regularisation. They are more frequent after pruning, because fewer features need less shrinkage.
- **Edge hits would not change the comparison.** The full model's wide-grid and authors'-grid means differ by only 0.11 pp.

---

## 7. Stage A: development check (6 development datasets × 5 splits; v1 and v2)

Stage A was run immediately before Stage B in the same session. Detach v2 was **not** changed after Stage A, so the pre-registration stayed valid.

**Mean over the 6 datasets (pp):**

| Budget | v1 − small_g | v2 − small_g | v2 − v1 |
|---|---|---|---|
| 0.0625 | +1.49 | +2.10 | +0.61 |
| 0.125 | −0.53 | −0.07 | +0.45 |
| 0.25 | −1.25 | −0.13 | +1.13 |
| 0.5 | −0.08 | +0.79 | +0.87 |

**Per dataset:**

- **MiddlePhalanxOutlineAgeGroup, the development-run outlier:**
  - v2 recovers **+3.0 to +4.7 pp over v1** at every budget.
  - It is still below `small_g`: −7.10 / −8.48 / −6.58 / −1.77 pp, against v1's −11.4 / −11.5 / −10.2 / −6.5 pp.
  - So stability averaging reduces the overfitting failure but does not remove it.
- **Worms:** v2 is better than v1 at 3 of 4 budgets (+2.34 pp at 0.125), and still slightly below `small_g`.
- **FiftyWords and WordSynonyms (many classes):** both versions are far ahead of `small_g`, for example +10.4 pp at 0.0625 on WordSynonyms. v2 and v1 are within ±0.5 pp of each other.
- **Herring:** both are ahead of `small_g`. v2 − v1 is mixed, from −2.2 to +1.6 pp.
- **ScreenType:** both are behind `small_g` by 1–3 pp, and v2 is about equal to v1.

**Takeaway.** On these 6 datasets v2 is better than v1 on average at every budget, with the gain mostly on the noisy datasets. Only 6 datasets were used, so this is supporting evidence, not a test.

---

## 8. Interpretation and caveats

1. **What the result supports.** On held-out datasets, importance-based pruning of a trained Hydra beats training a smaller Hydra from scratch **at small kernel budgets**. At 1/16 the difference is clearly significant and consistent across splits and metrics. At 1/8 it is borderline. At 1/4 and 1/2 there is no meaningful difference.

2. **What it does not show.**
   - **That stability averaging caused the success.** v1 was not run on the held-out data (`INCLUDE_DETACH_V1 = False`). The held-out effect at 0.0625 (+1.22 pp) is similar to v1's development effect (+1.07 pp). The rule may have passed because there were 72 datasets instead of 40, giving more statistical power, rather than because of v2. v2's improvement over v1 is supported only by the 6 Stage A datasets.
   - **That pruning beats keeping the full model.** Detach v2 is significantly below the full Hydra at every budget (−0.2 to −1.9 pp).

3. **Consistency with the development run.**

   | Budget | Development Δ (v1, 40 datasets) | Held-out Δ (v2, 72 datasets) |
   |---|---|---|
   | 0.5 | +0.28 | +0.13 |
   | 0.25 | +0.18 | +0.31 |
   | 0.125 | +0.57 | +0.56 |
   | 0.0625 | +1.07 | +1.22 |

   The two profiles are nearly identical: the effect grows as the budget shrinks. The advantage carried over to new data rather than coming from development choices.

4. **The failure mode still exists.**
   - 2 of 72 held-out datasets lose more than 5 pp: Earthquakes −8.35 pp at 0.125, and DistalPhalanxOutlineAgeGroup.
   - In Stage A, MiddlePhalanxOutlineAgeGroup is still −7 to −8.5 pp.
   - The losses cluster on noisy datasets with few classes.

5. **Practical trade-off.**
   - Detach v2 gives the inference cost of a 1/16-size model, at 1.2 pp higher accuracy than training that small model directly.
   - It pays about 3× the full model's training time, and up to 15 min per split on ElectricDevices.
   - The gain is largest with many classes (+2.5 pp with more than 10 classes) and about zero for binary problems.

6. **Limitations.**
   - **Splits are not independent.** The 5 splits are resamples of the same pooled data, so the statistics use one value per dataset.
   - **`small_g` seeds:** it is averaged over 3 kernel seeds, while Detach v2 uses the single full-model seed per split.
   - **Inference speed** of the pruned models was not timed directly in this notebook; see the decisive run.
   - **Timings** are wall-clock times on one machine with all cores.

---

## 9. Figures (`results/Shreed/figures/`)

- `detach_confirm_accuracy_vs_budget.png`: mean accuracy against kernel fraction, and Δ against the full model, for Detach v2 and `small_g`.
- `detach_confirm_per_dataset_delta.png`: per-dataset `detach_v2 − small_g` at each budget.
- `detach_confirm_kept_structure.png`: keep rate by relative dilation, and the diff(X) share.
- `detach_confirm_block_coverage.png`: block coverage against budget.
