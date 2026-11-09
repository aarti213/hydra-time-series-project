# Detach-Hydra Decisive Experiment: Full Results and Observations

**Notebook:** `notebooks/shreed_detach_hydra_decisive.ipynb` (full run, `QUICK_RUN = False`)
**Scale:** 40 development datasets × 5 splits (default split + 4 stratified resamples) = 200 dataset-splits, all completed (200/200, no failures). That is 9,200 model fits saved.
**Compute:**
- Experiment: 159.5 min on 10 threads.
- Timing: test-set transform on 6 datasets, 1 thread.

**Outputs** (all in `results/Shreed/`):
- CSVs: `detach_decisive_runs.csv`, `detach_decisive_paths.csv`, `detach_decisive_kept_groups.csv`, `detach_decisive_timing.csv`
- Auto-generated report: `DETACH_HYDRA_DECISIVE_FINDINGS.md`
- Figures: `figures/detach_decisive_*.png`

This file is my independent verification and write-up of that run. Every number below was either recomputed from the CSVs or cross-checked against the notebook output.

---

## 0. One-paragraph summary

At matched kernel budgets, **global Detach (importance-ranked group pruning) is consistently a little better than training a smaller Hydra (`small_g`)**:
- **Mean gain:** +0.28 / +0.18 / +0.57 / +1.07 pp at budgets 0.5 / 0.25 / 0.125 / 0.0625.
- **Wins:** about 2 wins per loss over 40 datasets at the two smallest budgets (26/2/12 and 26/3/11).
- **Significance:** after Holm correction it is **not significant** (Holm p = 0.09 at budgets 0.5, 0.125 and 0.0625; 0.14 at 0.25). So under the success rule fixed before the run, **the verdict is FAILURE, but a near miss.** The rule required a budget ≤ 0.25 with mean ≥ +0.5 pp, wins ≥ 2× losses and Holm p < 0.05. Budget 0.125 meets the mean and win conditions but not the p-value.
- **What holds it back:** much of the non-significance comes from **one dataset, MiddlePhalanxOutlineAgeGroup**, where Detach overfits the training data and loses about 11 pp.
- **Mechanism:** the gain comes from the **importance ranking**, which beats random pruning significantly at budgets 0.5 and 0.0625. It does **not** come from even coverage: forcing equal groups per block makes Detach *worse*.
- **Kernels:** fresh kernels (`small_g`) and a random subset of the big model's kernels are equivalent.
- **Speed:** speed-ups are proportional to the kernels kept, reaching 10.2× at 1/16 of the kernels.
- **Cost:** every pruned model is still significantly worse than full Hydra at budgets ≤ 0.25.

> **Correction to the auto-generated report.** In `results/Shreed/DETACH_HYDRA_DECISIVE_FINDINGS.md`, the line under the verdict, *"Importance pruning ≈ a smaller Hydra; even coverage across dilations and inputs matters more than the ranking"*, is a **fixed sentence** that the verdict cell prints in every FAILURE case. It is not computed from the data, and the data contradict its second half (see §5). Use the interpretation in this file instead. The tables in that report are correct.

---

## 1. The question

Hydra's cost is proportional to its number of kernels. If we want a model that is 2–16× cheaper, are we better off:
- **(a)** training the full Hydra and **pruning** the least important kernel groups (Detach), or
- **(b)** simply training a **smaller Hydra** (`small_g`: fewer groups per dilation) with exactly the same number of kernels?

If (a) is not better than (b), pruning gives no benefit beyond "use a smaller g".

## 2. Setup (as implemented)

**Full model.** Official Hydra (`hydra-authors-repo/code/hydra.py`), k = 8 kernels per group, g = 64 groups, kernel seed = split id.
- **Structure:** for every (dilation, input ∈ {raw X, diff X}) "block" there are 32 groups, so G = num_dilations × 64 groups in total.
- **Features:** each group gives 16 features (8 soft-max counts + 8 hard-min counts).

**Budgets.** Keep 0.5 / 0.25 / 0.125 / 0.0625 of the groups. `small_g` matches each budget exactly with g′ = 32 / 16 / 8 / 4. Group counts, kernel counts and feature counts are identical between all methods at a budget; this was asserted for every row.

**Classifier (all models).** `SparseScaler` fitted on train only, then `RidgeClassifierCV(alphas = logspace(-6, 6, 25))` with leave-one-out alpha selection on train only. Full Hydra is also fitted with the authors' grid `logspace(-3, 3, 10)` for reference.

**Methods.**

| method | what it keeps | draws |
|---|---|---|
| `full` | all groups | 1 |
| `detach` | Detach path (train data only): fit ridge → score each group by the L2 norm of its coefficients → drop the weakest 10% → re-select alpha → repeat. Removals are clamped so the path lands exactly on each budget. Keeps whatever survives, so whole blocks may be emptied | 1 |
| `detach_stratified` | Same importance ordering, but forced to keep exactly 32·b groups in **every** block | 1 |
| `random` | random groups chosen globally | 3 |
| `random_stratified` | random groups, 32·b per block (the same layout as `small_g`) | 3 |
| `small_g` | a freshly drawn Hydra with g′ = 64·b | 3 seeds |

Pruned models are evaluated by subsetting the full model's feature columns. This is **bit-exact** with physically rebuilding a pruned Hydra: `torch.equal` on train and test was asserted for every dataset and again in the timing section.

**Statistics.**
- Per dataset: average over the 5 splits (and over the 3 draws/seeds).
- Then over the 40 datasets: mean Δ in pp, a 95% bootstrap CI, W/D/L (|Δ| < 0.05 pp counts as a draw), a Wilcoxon signed-rank test, and Holm correction within each comparison family (4 budgets).

**Datasets (40).** Beef, BirdChicken, Car, CricketX/Y/Z, DistalPhalanxTW, ECG5000, FiftyWords, Fish, FordA, FordB, Haptics, Herring, InsectWingbeatSound, ItalyPowerDemand, LargeKitchenAppliances, Lightning7, Meat, MedicalImages, MiddlePhalanxOutlineAgeGroup, MiddlePhalanxOutlineCorrect, OSULeaf, OliveOil, Phoneme, Plane, ProximalPhalanxOutlineCorrect, ProximalPhalanxTW, ScreenType, ShapeletSim, Strawberry, SwedishLeaf, SyntheticControl, ToeSegmentation1, Trace, UWaveGestureLibraryY, Wafer, WordSynonyms, Worms, Yoga.

## 3. Verification: is the run trustworthy?

| check | result |
|---|---|
| Official Hydra arithmetic (`divisor = 2`, `h = g // 2`, groups = dilations × g, dilations independent of g) | PASS |
| Pilot Test 1: PrunedHydra with all groups == official Hydra (ECG200, Beef) | PASS |
| Pilot Tests 2+3: 14 random/structured masks, rebuilt model == kept columns; feature/kernel counts | PASS |
| Pilot Test 4: scaling once then subsetting == refitting the scaler | PASS (max diff 0.0) |
| Same seed → identical features; different seed → different | PASS |
| Pipeline reproduces Notebook 1 (ECG200 seed 0 = 0.8700) | PASS |
| All 40 datasets load; split sizes, class counts and no train/test overlap asserted for all 5 splits | PASS |
| `detach_path` receives training arrays only (no test data can influence pruning or alpha) | PASS |
| Rebuilt PrunedHydra == kept columns (detach + detach_stratified, budget 0.25, every dataset) | PASS |
| Exact matched budget sizes in every pair | PASS |
| **My recheck:** all 9,200 rows have `groups_kept == budget × 64 × dilations` and `n_features == 16 × groups` | PASS |
| **My recheck:** recomputed means, medians and raw Wilcoxon p from the raw CSV | match (p differs at the 3rd decimal because of draw-tolerance handling; no conclusion changes) |
| Timing: rebuilt pruned models from the saved masks == kept columns, every timed model | PASS |
| Failures | none (no failures CSV written) |

**Conclusion: the implementation is correct and the comparison is fair.** Group, kernel and feature counts are identical, every method uses the same classifier, and pruning uses no test data.

## 4. Main results

### 4.1 Mean accuracy over 40 datasets

| budget | full | **detach** | detach_stratified | random | random_stratified | small_g |
|---|---|---|---|---|---|---|
| 0.5 | 0.8418 | **0.8403** | 0.8378 | 0.8382 | 0.8380 | 0.8374 |
| 0.25 | 0.8418 | **0.8334** | 0.8311 | 0.8317 | 0.8314 | 0.8316 |
| 0.125 | 0.8418 | **0.8283** | 0.8249 | 0.8222 | 0.8226 | 0.8226 |
| 0.0625 | 0.8418 | **0.8206** | 0.8151 | 0.8058 | 0.8094 | 0.8098 |

Full Hydra: 0.8418 with the wide alpha grid, 0.8436 with the authors' grid.

**Mean Δ vs full Hydra (pp):**

| budget | detach | detach_stratified | random | random_stratified | small_g |
|---|---|---|---|---|---|
| 0.5 | **−0.15** | −0.40 | −0.36 | −0.38 | −0.44 |
| 0.25 | **−0.84** | −1.07 | −1.01 | −1.04 | −1.02 |
| 0.125 | **−1.35** | −1.69 | −1.96 | −1.92 | −1.92 |
| 0.0625 | **−2.12** | −2.67 | −3.60 | −3.24 | −3.20 |

**Observations:**
- **Detach has the highest mean accuracy at every budget.** Its lead over the controls grows as the budget shrinks.
- **The four non-Detach methods are nearly indistinguishable** at 0.5–0.125 (within about 0.06 pp of each other). Global `random` falls behind only at 0.0625.
- **Detach vs full Hydra:**
  - At 0.5, detach costs only −0.15 pp. That is not significantly worse than full (15 wins / 18 losses, p = 0.39), at about half the kernels.
  - At ≤ 0.25 detach is **significantly worse** than full: 8/28 (p = 0.011), 8/29 (p = 0.001) and 5/31 (p < 0.001) wins/losses at 0.25 / 0.125 / 0.0625.

### 4.2 The decisive comparison: detach vs small_g (matched budgets)

| comparison @ budget | mean (pp) | 95% CI | median | W/D/L | p | **Holm p** |
|---|---|---|---|---|---|---|
| detach − small_g @ 0.5 | +0.28 | [−0.19, +0.72] | +0.12 | 22/6/12 | 0.032 | 0.090 |
| detach − small_g @ 0.25 | +0.18 | [−0.55, +0.78] | +0.10 | 22/5/13 | 0.141 | 0.141 |
| detach − small_g @ 0.125 | **+0.57** | [−0.35, +1.40] | +0.37 | **26/2/12** | 0.022 | 0.090 |
| detach − small_g @ 0.0625 | **+1.07** | [+0.01, +2.08] | +0.54 | **26/3/11** | 0.023 | 0.090 |
| detach_stratified − small_g @ 0.5 | +0.03 | [−0.43, +0.46] | +0.04 | 20/4/16 | 0.619 | 1.000 |
| detach_stratified − small_g @ 0.25 | −0.05 | [−0.62, +0.43] | +0.00 | 16/8/16 | 0.922 | 1.000 |
| detach_stratified − small_g @ 0.125 | +0.23 | [−0.38, +0.70] | +0.20 | 26/3/11 | 0.015 | 0.061 |
| detach_stratified − small_g @ 0.0625 | +0.53 | [−0.02, +1.10] | +0.28 | 26/4/10 | 0.027 | 0.081 |

**Verdict under the pre-registered rule: FAILURE (a near miss).**
- **Budget 0.125:** mean +0.57 pp ≥ 0.5 ✓, W/L 26/12 ≥ 2× ✓, Holm p 0.090 ✗.
- **Budget 0.0625:** outside the "≤ 0.25 with all conditions" window only because of p. Its mean is +1.07 pp ✓, W/L 26/11 ✓, Holm p 0.090 ✗.
- **The sign is consistent:** detach − small_g is positive at all 4 budgets, the median is positive at all 4, and the raw (uncorrected) p < 0.05 at 3 of 4 budgets.

### 4.3 Per-dataset picture and the MiddlePhalanxOutlineAgeGroup outlier

The detach − small_g gains are spread across many datasets, but a few large values on each side dominate the mean.

| budget | 5 worst datasets (pp) | 5 best datasets (pp) |
|---|---|---|
| 0.25 | MiddlePhalanxOutlineAgeGroup −10.2, Worms −3.1, ScreenType −2.1, MiddlePhalanxOutlineCorrect −1.4, DistalPhalanxTW −1.3 | FiftyWords +2.0, ShapeletSim +2.3, Car +2.7, Beef +2.7, WordSynonyms +4.6 |
| 0.125 | MiddlePhalanxOutlineAgeGroup −11.5, Worms −4.3, ScreenType −2.8, ProximalPhalanxTW −1.3, LargeKitchenAppliances −0.7 | OliveOil +3.3, ShapeletSim +3.4, Lightning7 +3.8, Herring +4.5, WordSynonyms +7.9 |
| 0.0625 | MiddlePhalanxOutlineAgeGroup −11.4, DistalPhalanxTW −2.3, Worms −2.2, ScreenType −1.5, OliveOil −1.3 | Beef +4.9, CricketZ +5.3, FiftyWords +6.1, Herring +7.4, WordSynonyms +10.5 |

**MiddlePhalanxOutlineAgeGroup is a clear failure mode of Detach: it overfits the training data.**

| method (accuracy) | 0.5 | 0.25 | 0.125 | 0.0625 |
|---|---|---|---|---|
| detach | 0.636 | 0.608 | 0.596 | 0.599 |
| detach_stratified | 0.642 | 0.626 | 0.621 | 0.661 |
| random / random_strat / small_g | ≈ 0.70–0.71 | ≈ 0.71 | ≈ 0.71 | ≈ 0.71 |
| full | 0.708 | | | |

- **The training score improves while test accuracy falls.** Along the Detach path (split 0), the train leave-one-out error keeps improving as groups are removed: GCV score −0.364 → −0.288 at 58 groups. Meanwhile test accuracy drops by about 11 pp.
- **Why:** pruning picks the groups that fit the (noisy, 3-class, 400-series) training labels best, and those don't generalise.
- **Random subsets don't have this problem**, so the issue is *selection on training data*, not having fewer kernels.

**Sensitivity check (exploratory, decided after seeing the data, so not a valid replacement for the pre-registered test).** Dropping only MiddlePhalanxOutlineAgeGroup (39 datasets) gives:

| budget | mean (pp) | Wilcoxon p | Holm p |
|---|---|---|---|
| 0.5 | +0.46 | 0.011 | 0.027 |
| 0.25 | +0.44 | 0.071 | 0.071 |
| 0.125 | +0.88 | 0.007 | 0.027 |
| 0.0625 | +1.39 | 0.008 | 0.027 |

So one dataset is enough to move the result across the significance line. The honest reading: **Detach is usually somewhat better than small_g, but it can fail badly when its training-set selection overfits.**

**Noise level, for context:** the average best-minus-worst spread across the 3 `small_g` seeds is 1.6 / 1.6 / 2.0 / 2.7 pp (budgets 0.5 → 0.0625). Across the 3 `random` draws it is 1.3 / 1.6 / 2.1 / 3.0 pp. Single-split differences of 1–3 pp are within seed noise; only the 40-dataset aggregate is informative.

### 4.4 Breakdowns of Δ(detach − small_g), pp (descriptive; small groups)

| group | n | 0.5 | 0.25 | 0.125 | 0.0625 |
|---|---|---|---|---|---|
| train ≤ 100 | 11 | +0.69 | +0.87 | +1.64 | +1.61 |
| train 101–500 | 22 | +0.12 | −0.10 | +0.17 | +0.87 |
| train > 500 | 7 | +0.15 | −0.06 | +0.15 | +0.85 |
| length ≤ 200 | 12 | −0.49 | −1.02 | −0.66 | −0.40 |
| length 201–500 | 19 | +0.69 | +1.02 | +1.60 | +2.13 |
| length > 500 | 9 | +0.46 | −0.02 | +0.05 | +0.80 |
| 2 classes | 12 | +0.52 | +0.13 | +0.73 | +1.18 |
| 3–10 classes | 20 | −0.04 | −0.32 | −0.27 | −0.18 |
| > 10 classes | 8 | +0.75 | +1.48 | +2.46 | +4.04 |

What the breakdowns suggest:
- **Many classes (> 10)** is where Detach helps most: WordSynonyms, FiftyWords, the Cricket datasets. A small random subset probably loses class-specific detectors, while the ranking keeps them.
- **Length 201–500** also gains a lot. **Short series (≤ 200)** lose, a group that includes the Phalanx datasets.
- **3–10 classes** is slightly negative. It contains MiddlePhalanxOutlineAgeGroup and Worms.
- By type: Simulated (+1.5 to +3.5, n = 2) and Motion (+0.4 to +1.8, n = 7) gain; Device (n = 2) loses.

These are descriptive only: the groups are small and not tested.

## 5. Mechanism: why does Detach win when it does?

| comparison (what it isolates) | 0.5 | 0.25 | 0.125 | 0.0625 |
|---|---|---|---|---|
| **detach − random** (does the importance ranking help?) | **+0.21 (28/4/8) ✓** | +0.16 (24/5/11) | +0.61 (24/4/12) | **+1.48 (30/1/9) ✓** |
| **detach_stratified − detach** (does forcing even block coverage help Detach?) | **−0.25 (9/5/26) ✓, worse** | −0.22 (14/4/22) | −0.34 (14/6/20) | −0.54 (15/3/22) |
| detach_stratified − random_stratified (ranking within blocks) | −0.02 | −0.02 | +0.22 (23/3/14) | **+0.57 (25/6/9) ✓** |
| random_stratified − random (even coverage, no ranking) | −0.02 | −0.04 | +0.04 | **+0.36 (24/7/9) ✓** |
| small_g − random_stratified (fresh kernels vs a subset of the big model) | −0.06 | +0.02 | −0.01 | +0.04 |

✓ = Holm p < 0.05. Holm p values: detach − random 0.019 and 0.003; detach_stratified − detach 0.025; detach_stratified − random_stratified 0.007; random_stratified − random 0.025.

**Observations:**
1. **The ranking is what helps.** Detach beats random pruning at every budget, significantly at 0.5 and 0.0625. At 0.0625 it wins on 30 of 40 datasets (+1.48 pp).
2. **Forcing even coverage hurts Detach.** Its *uneven* allocation (emptying some blocks and concentrating on others, see §6) is part of why it works. Making it keep equal groups per block loses 0.2–0.5 pp at every budget, significantly at 0.5. **This contradicts the fixed headline in the auto-generated report.**
3. **Coverage only matters when pruning is random and extreme.** At 1/16 of the kernels, random global pruning empties blocks by chance (88% coverage) and loses 0.36 pp against random-with-even-coverage. At larger budgets coverage makes no difference.
4. **Fresh kernels vs reused kernels makes no difference.** `small_g` (a new smaller Hydra) ≈ `random_stratified` (a random subset of the big Hydra with the same layout) at every budget, all |Δ| ≤ 0.06 pp. So a "smaller Hydra" is just a random subset. Detach's advantage over `small_g` is entirely the advantage of *choosing* kernels over picking them at random.
5. **Ranking within blocks helps only at tight budgets.** With the layout fixed, the ranking adds +0.57 pp at 0.0625 (significant) and +0.22 at 0.125, but nothing at 0.5–0.25.

## 6. What Detach keeps

**Keep rate by relative dilation (detach; 1.0 = uniform, as in small_g):**

| budget | short | mid | long |
|---|---|---|---|
| 0.5 | 0.87 | 1.02 | 1.15 |
| 0.25 | 0.82 | 1.00 | 1.22 |
| 0.125 | 0.75 | 1.02 | 1.28 |
| 0.0625 | **0.69** | 1.04 | **1.33** |

There is a clear, systematic preference for **long dilations** (wide receptive fields) and against short ones, and it gets stronger as the budget shrinks.

**Share of kept groups on diff(X) (0.5 = uniform):** 0.48 → 0.45 → 0.40 → 0.36 (budget 0.5 → 0.0625). The range across datasets is wide: at 0.0625 it goes from 0.00 (raw only) to 0.83 (mostly diff). On average Detach favours the **raw series**, but which input matters is dataset-specific.

**Block coverage (fraction of (dilation, input) blocks that keep ≥ 1 group):**

| method | 0.5 | 0.25 | 0.125 | 0.0625 |
|---|---|---|---|---|
| detach | 0.997 | 0.961 | 0.881 | **0.720** (min 0.33) |
| random | 1.000 | 1.000 | 0.989 | 0.884 |
| stratified / small_g | 1.000 | 1.000 | 1.000 | 1.000 |

At 1/16 of the kernels Detach drops more than a quarter of the blocks entirely, and in the most extreme case two thirds of them. It is deliberately concentrating capacity.

**Consistency across splits (weighted Jaccard of per-block counts; random = chance level):**

| budget | detach | random |
|---|---|---|
| 0.5 | 0.842 | 0.828 |
| 0.25 | 0.748 | 0.724 |
| 0.125 | 0.676 | 0.600 |
| 0.0625 | **0.606** | 0.477 |

At small budgets Detach picks a **noticeably more reproducible** block structure than chance (0.61 vs 0.48 at 0.0625). It is finding real, dataset-specific structure, not noise. At large budgets the difference is small.

## 7. Speed (1 thread, test-set transform, median of 6 timing datasets: Wafer, FordA, Phoneme, Yoga, ECG5000, SwedishLeaf)

| budget | detach | detach_stratified | small_g | kernel fraction |
|---|---|---|---|---|
| 1.0 (full) | 1.00× (5.78 ms/series) | | | 1.0 |
| 0.5 | 1.91× (3.27 ms) | 1.90× (3.05 ms) | 1.91× (3.05 ms) | 0.5 |
| 0.25 | 3.54× (1.77 ms) | 3.41× (1.71 ms) | 3.45× (1.70 ms) | 0.25 |
| 0.125 | 6.32× (0.97 ms) | 5.95× (0.97 ms) | 5.94× (0.97 ms) | 0.125 |
| 0.0625 | **10.24×** (0.57 ms) | 9.02× (0.64 ms) | 9.03× (0.64 ms) | 0.0625 |

Full test-set transform times: Phoneme 49.4 s, Yoga 24.9 s, Wafer 20.1 s, FordA 15.5 s, ECG5000 12.2 s, SwedishLeaf 1.3 s.

**Observations:**
- **Speed-up is close to proportional to the kernels kept** (about 95% of ideal at 0.5, about 57–64% at 0.0625, where fixed overheads dominate).
- **At equal budget, `small_g` and `detach_stratified` run at identical speed** (identical structure).
- **Global `detach` is slightly faster at small budgets** (10.2× vs 9.0×) because it empties whole dilation blocks, so those convolutions are skipped entirely.
- **The useful trade-off:**
  - Detach at 0.0625 reaches **0.8206 at 10.2×**, essentially the same accuracy as small_g at 0.125 (**0.8226 at 5.9×**). So Detach gives about the same accuracy at **≈ 1.7× more speed** (−0.2 pp).
  - Likewise detach at 0.125 (0.8283, 6.3×) is only 0.33 pp below small_g at 0.25 (0.8316, 3.45×), at 1.8× the speed.

## 8. Alpha (a lesson for reproduction, not part of the decision)

- **Grid edge.** With the authors' grid `logspace(-3, 3, 10)`, full Hydra's selected alpha sits **at the grid edge in 74% of fits** (61 fits at 0.001, 87 at 1000). With the wide grid `logspace(-6, 6, 25)` that falls to 16.5%; the remaining edge hits are mostly at 1e-6 on tiny-training-set datasets, where the LOO curve is flat.
- **Accuracy.** Despite this, accuracy hardly changes: wide − authors' grid = −0.18 pp (95% CI [−0.63, +0.25]), W/D/L 11/17/12, p = 0.51. The authors' narrow grid is a reproduction detail, not an accuracy problem.
- **Along the path.** During the Detach path, alpha is at the wide-grid edge in 20.5% of steps.

## 9. What the figures show

- **`detach_decisive_accuracy_vs_speed.png`**
  - *Left (accuracy vs measured speed-up, log x-axis):* the **blue detach curve lies above all others at every speed-up**. The gap widens to the right (0.8206 vs about 0.810 at about 9–10×). The other four curves are bunched together from 1.9× to 6×, and only `detach_stratified` (orange) separates upward at 0.125 and 0.0625. Global `random` (green) is lowest at the far right.
  - *Right (Δ vs full with 95% CIs):* all methods lose accuracy roughly linearly in log(budget). Detach's points sit highest at every budget. Only detach at 0.5 (CI reaching +0.3) and detach_stratified at 0.5 overlap zero. The CIs of the different methods overlap substantially at every budget, which matches the non-significant Holm results.
- **`detach_decisive_cd_budget_0.25.png`** (critical-difference diagram). Mean ranks: full 2.06, **detach 3.21**, small_g 3.70, detach_stratified 3.80, random_stratified 3.98, random 4.25. **Full Hydra stands alone** (significantly better than every pruned model). One bar joins all five pruned methods: no significant difference among them, although detach has the best rank.
- **`detach_decisive_cd_budget_0.125.png`.** full 1.82, **detach 3.19**, detach_stratified 3.51, random_stratified 4.03, small_g 4.20, random 4.25. Same structure: full alone, the pruned methods joined by one bar, detach ranked best of them and both Detach variants ahead of all three controls.
- **`detach_decisive_kept_groups.png`**
  - *Left (heatmap, detach at budget 0.25, keep rate ÷ uniform per dataset × (input, short/mid/long dilation)):* the dominant pattern is **dark red in the "raw long" column** (≥ 2× uniform) for many datasets: CricketX, CricketZ, FiftyWords, Haptics, MedicalImages, Phoneme, SyntheticControl, UWaveGestureLibraryY, Wafer, WordSynonyms, Worms. The short-dilation columns are mostly blue. There are exceptions (Beef, BirdChicken, ShapeletSim, LargeKitchenAppliances, ScreenType keep raw-short heavily). ItalyPowerDemand shows white (empty) mid cells because, with length 24, it has too few dilations to fill a "mid" bin.
  - *Right (block coverage vs budget):* detach's coverage falls steeply (0.997 → 0.961 → 0.881 → 0.720), much faster than random (0.884 at 0.0625). The stratified methods and small_g stay at 1.0.
- **`detach_decisive_per_dataset_delta.png`** (detach − small_g per dataset, 4 bars per dataset). Most bars are small and positive. **The largest positive bars are WordSynonyms (+10.5 pp at 0.0625), Herring (+7.4), FiftyWords (+6.1), CricketZ (+5.3), Beef (+4.9).** Bars often grow as the budget shrinks. **The largest negative bar by far is MiddlePhalanxOutlineAgeGroup** (−6.5 to −11.5 pp at all budgets), followed by Worms and ScreenType (−2 to −4 pp). Plane, Trace and Wafer are at about 0 (at ceiling).

## 10. Interpretation and conclusions

1. **Main question.** Under the pre-registered criterion, **importance pruning does not significantly beat a smaller Hydra** (Holm p ≈ 0.09). **But the direction is consistent:** positive at every budget, about 2:1 wins at small budgets, +1.07 pp at 1/16 of the kernels. The effect grows as the budget shrinks.
2. **The failure mode is overfitting during selection.** Ranking groups by ridge coefficients fit on the same training data can choose groups that fit training noise (MiddlePhalanxOutlineAgeGroup: −11 pp while the training LOO score improves). Without that one dataset the result would be significant. This points to a concrete fix (§11).
3. **"Smaller Hydra" = random subset.** `small_g` is statistically identical to randomly subsetting the big model. So anything Detach gains over `small_g` comes from *choosing* kernels.
4. **Uneven allocation is a feature, not a bug.** Detach shifts capacity to long dilations and the raw series, empties less useful blocks, and does so more reproducibly than chance. Forcing even coverage removes part of the benefit.
5. **Practical value.** For deployment under a tight latency budget, Detach at 1/16 of the kernels gives about the accuracy of small_g at 1/8 of the kernels, at about 1.7× the speed (10.2× vs 5.9× over full Hydra). The price is an extra training cost (the pruning path) and the risk of the occasional overfitting dataset.
6. **No free lunch.** At ≤ 1/4 of the kernels, every pruned model is significantly less accurate than full Hydra (detach −0.84 to −2.12 pp). Only at 1/2 is detach statistically tied with full (−0.15 pp, p = 0.39).

## 11. Suggested next steps (not run)

- **Guard against overfitting the selection.** Stop pruning, or pick the budget, using a held-out validation split or cross-validation *inside* the training set, rather than the same training data used for scoring. Or score groups with an inner CV / stability selection across bootstrap resamples. MiddlePhalanxOutlineAgeGroup is the test case.
- **Use more splits** (for example 10–30 instead of 5) to tighten the CIs. The Holm p ≈ 0.09 is close enough that more splits could settle it either way.
- **Keep global allocation** (not stratified). If blocks must be kept, use a soft minimum (for example ≥ 1 group per block) instead of an equal count.

## 12. Caveats

- **Splits:** 5 splits per dataset (the default + 4 stratified resamples), not the 30 used in the Hydra paper.
- **Dataset choice:** these 40 are the "development" datasets the Hydra authors also used for tuning. The list comes from the team's extraction of the ROCKET appendix and couldn't be re-verified against the PDF.
- **Timing:** one machine, one thread, 6 datasets; the speed-ups are relative, not absolute.
- **Pruning step:** the Detach step is 10% per iteration. A finer step could change the selection slightly.
- **Post-hoc analyses:** the analysis excluding MiddlePhalanxOutlineAgeGroup and the dataset-property breakdowns were done after seeing the data; treat them as hypotheses.
