# Detach-Hydra Decisive Experiment: Verification and Findings (QUICK_RUN)

> **Important: this is the QUICK_RUN, not the decisive experiment.** The notebook ran with `QUICK_RUN = True`: 3 datasets (Beef, SwedishLeaf, Wafer) × 2 resamples = 6 pairs. With n = 3 datasets no comparison can reach significance. The run shows that **the pipeline is correct and ready**. It does **not** say whether Detach beats `small_g`. That needs `QUICK_RUN = False` (40 datasets × 5 resamples).

**Notebook:** `notebooks/shreed_detach_hydra_decisive.ipynb`
**Outputs from this run** (all in `results/Shreed/`):
- `detach_decisive_quick_runs.csv`: 276 rows
- `detach_decisive_quick_paths.csv`: 188 path steps
- `detach_decisive_quick_kept_groups.csv`: 15,360 rows
- `detach_decisive_quick_timing.csv`: 26 rows
- 5 figures in `results/Shreed/figures/detach_decisive_quick_*.png`

**Missing output:** the notebook's auto-generated report, `results/Shreed/DETACH_HYDRA_DECISIVE_FINDINGS_QUICK.md`, was **not written**. Either the last cell ("write the findings report") wasn't run, or it stopped before writing. The saved notebook has no cell outputs, so I can't tell which. This file covers the same content, recomputed independently from the CSVs.

---

## 1. Question

At matched kernel budgets, does importance-ranked group pruning (Detach-Hydra) beat simply training a smaller Hydra (`small_g`) with the same number of kernels?

## 2. Setup (as implemented)

- **Full model.** Official Hydra (`hydra-authors-repo/code/hydra.py`), k = 8, g = 64, kernel seed = resample id. Groups G = num_dilations × g: Beef 384, SwedishLeaf 256, Wafer 320. Each group is 8 kernels, giving 16 features (max part + min part).
- **Budgets.** 0.5 / 0.25 / 0.125 / 0.0625 of the groups. These are matched exactly by `small_g` with g′ = 32 / 16 / 8 / 4, so the total group and kernel counts are identical.
- **Classifier.** Every model uses `SparseScaler` (fit on train only) then `RidgeClassifierCV(alphas = logspace(-6, 6, 25))`. The full model is also fitted with the authors' grid `logspace(-3, 3, 10)` for reference.
- **Methods:**

| method | what it keeps |
|---|---|
| `full` | all groups (wide alpha grid) |
| `full_author_grid` | all groups, authors' alpha grid |
| `detach` | the active groups at the budget step of the Detach path. Train only, alpha re-selected at every step, 10% removed per step, clamped to land exactly on each budget. Score = L2 norm of a group's ridge coefficients |
| `detach_stratified` | the top 32·b groups **in every (dilation, input) block**, under the Detach importance ordering |
| `random` | uniformly random groups, global (3 draws) |
| `random_stratified` | random, 32·b per block (3 draws) |
| `small_g` | a fresh Hydra with g′ = 64·b (3 kernel seeds) |

- **How pruned accuracy is computed.** Pruned models are evaluated by column-subsetting the full model's features. This is bit-exact with rebuilding a pruned transform (asserted with `torch.equal`, see §3).
- **Success rule** (set before the run): at some budget ≤ 0.25, detach − small_g must be ≥ +0.5 pp mean, with wins ≥ 2× losses and Holm p < 0.05.

## 3. Verification

I checked this independently by re-reading the CSVs. I did not re-run the notebook.

| check | result |
|---|---|
| Official Hydra arithmetic: `divisor = 2`, `h = g // 2`, groups = num_dilations × g, dilations independent of g | **PASS** (notebook check) |
| Pilot Tests 1–4 (`group_feature_index`, `kept_columns`, `PrunedHydra` equivalence, max-only), copied word for word | **PASS** (the notebook stops on any failure, and it reached the end) |
| All datasets load; resample sizes, class counts and no train/test overlap asserted | **PASS** |
| `detach_path` receives training arrays only (signature check; no test array parameter) | **PASS** |
| Rebuilt `PrunedHydra` == kept columns (`torch.equal`, train and test) at budget 0.25 | **PASS** (an assertion inside `run_pair`; all 6 pairs completed) |
| Timing section rebuilds masks from the kept-groups CSV and asserts equality | **PASS** (timing CSV written) |
| **Matched budgets:** every non-full row has `groups_kept == budget × 64 × num_dilations` | **PASS**: all 264 rows (my recomputation) |
| `detach_stratified` keeps an equal count in every block | **PASS**: all 24 rows |
| Kept-groups CSV agrees with the per-block counts in the runs CSV | **PASS**: 48/48 rows, 0 mismatches (my recomputation) |
| Random draws are really different | **PASS**: 3 distinct block layouts per (dataset, resample, budget) |
| **Reproducibility:** your run vs my earlier smoke run of the same notebook | **PASS: bit-identical.** All 276 rows match, max accuracy difference 0.0, all alphas identical |
| Path lands on every budget | **PASS**: `budget_reached` is set exactly 4× per pair (steps 7/14/22/30, or 31 for SwedishLeaf) |

**One oddity I looked into (harmless).** On Beef, `full` picks alpha = 3.2e-5, but step 0 of the Detach path (same features, columns in group order) picks 1e-6. I reproduced this outside the notebook. With n = 30 < p = 6144, the leave-one-out score is flat to about 3e-8 over alphas 1e-6 to 1e-4, so which alpha wins comes down to float rounding from the column order. All of those alphas give the same interpolating solution. It is not a bug, and it doesn't affect accuracy.

## 4. Results (n = 3 datasets: NOT conclusive)

### 4.1 Mean accuracy over datasets

| budget | full | detach | detach_stratified | random | random_stratified | small_g |
|---|---|---|---|---|---|---|
| 0.5 | 0.9251 | 0.9234 | **0.9295** | 0.9248 | 0.9251 | 0.9235 |
| 0.25 | 0.9251 | 0.9258 | **0.9319** | 0.9279 | 0.9212 | 0.9175 |
| 0.125 | 0.9251 | 0.9048 | **0.9291** | 0.9070 | 0.9124 | 0.9144 |
| 0.0625 | 0.9251 | **0.9178** | 0.9099 | 0.8958 | 0.9011 | 0.8892 |

Full Hydra: wide grid 0.9251, authors' grid 0.9253.

**Mean Δ vs full Hydra (pp):**

| budget | detach | detach_stratified | random | random_stratified | small_g |
|---|---|---|---|---|---|
| 0.5 | −0.16 | +0.45 | −0.03 | +0.01 | −0.16 |
| 0.25 | +0.07 | +0.68 | +0.29 | −0.39 | −0.75 |
| 0.125 | −2.02 | +0.41 | −1.80 | −1.27 | −1.06 |
| 0.0625 | −0.72 | −1.52 | −2.93 | −2.40 | −3.58 |

### 4.2 Decisive comparison (paired per dataset, mean over the 2 resamples)

| comparison @ budget | mean (pp) | 95% bootstrap CI | W/D/L | Holm p |
|---|---|---|---|---|
| detach − small_g @ 0.5 | −0.00 | [−0.01, +0.00] | 0/3/0 | 1.00 |
| detach − small_g @ 0.25 | +0.82 | [−0.29, +2.78] | 1/1/1 | 1.00 |
| detach − small_g @ 0.125 | −0.96 | [−2.78, +0.05] | 1/0/2 | 1.00 |
| detach − small_g @ 0.0625 | +2.86 | [−0.21, +8.33] | 2/0/1 | 1.00 |
| detach_stratified − small_g @ 0.5 | +0.61 | [−0.01, +1.67] | 2/1/0 | 1.00 |
| detach_stratified − small_g @ 0.25 | +1.44 | [−0.13, +4.44] | 1/1/1 | 1.00 |
| detach_stratified − small_g @ 0.125 | +1.47 | [−0.01, +3.89] | 2/1/0 | 1.00 |
| detach_stratified − small_g @ 0.0625 | +2.06 | [−0.06, +5.00] | 2/0/1 | 1.00 |

**Per dataset, detach − small_g (pp, mean over resamples):**

| dataset | 0.5 | 0.25 | 0.125 | 0.0625 |
|---|---|---|---|---|
| Beef (30 train, 5 classes) | 0.00 | +2.78 | −2.78 | **+8.33** |
| SwedishLeaf (500 train, 15 classes) | 0.00 | −0.29 | +0.05 | +0.45 |
| Wafer (1000 train, 2 classes) | −0.01 | −0.02 | −0.16 | −0.21 |

**Almost all of the signal comes from Beef.** With 30 test series, one test series is worth 3.33 pp, so +8.33 pp is about 2.5 series on average. SwedishLeaf and Wafer differences are within ±0.5 pp. Wafer is at ceiling (full = 1.000).

### 4.3 Mechanism comparisons (all Holm p = 1.00)

| comparison | 0.5 | 0.25 | 0.125 | 0.0625 |
|---|---|---|---|---|
| detach_stratified − detach (forced block coverage) | +0.61 (2/1/0) | +0.61 (2/1/0) | **+2.43 (3/0/0)** | −0.80 (2/0/1) |
| detach_stratified − random_stratified (ranking within blocks) | +0.44 (2/1/0) | +1.08 (1/1/1) | +1.68 (2/1/0) | +0.88 (2/0/1) |
| detach − random (global ranking) | −0.13 | −0.22 | −0.22 | +2.20 |
| random_stratified − random (even coverage alone) | +0.04 | −0.68 | +0.53 | +0.53 |
| small_g − random_stratified (fresh kernels vs subset) | −0.17 | −0.36 | +0.20 | −1.18 |

### 4.4 Timing (1 thread, test-set transform; median over the timing datasets)

QUICK_RUN times only 2 of the 6 timing datasets (Wafer, SwedishLeaf), the ones that are in the quick list.

| budget | detach speed-up | detach_stratified | small_g | kernel fraction |
|---|---|---|---|---|
| 0.5 | 1.89× | 1.89× | 1.90× | 0.5 |
| 0.25 | 3.51× | 3.41× | 3.42× | 0.25 |
| 0.125 | 6.03× | 5.70× | 5.72× | 0.125 |
| 0.0625 | 9.60× | 8.59× | 8.70× | 0.0625 |

Full Hydra latency: 2.74 ms per series (Wafer 3.21 ms, SwedishLeaf 2.28 ms).
- **Speed-up is close to proportional to kernels kept.** It falls short at small budgets because of fixed overhead.
- Global `detach` is slightly faster at 0.0625 (9.6× vs about 8.6×) because it empties whole dilation blocks, which skips those convolutions entirely.
- At equal budget, `small_g` and `detach_stratified` cost the same. The structure is identical (same count per block).

### 4.5 Kept groups (what Detach keeps)

- **Block coverage.** At 0.0625, global `detach` keeps at least one group in only **81%** of (dilation, input) blocks on average (minimum 60%); at 0.125 it is 97.9%. Global random keeps 92.9% and 98.9%. The stratified methods and `small_g` are 100% by construction.
- **Dilation preference (detach, keep rate ÷ uniform).** At 0.0625: short 0.92, mid 0.96, long 1.27. At 0.25: 1.02, 1.01, 1.11. There is a mild lean toward long dilations as the budget shrinks.
- **Input preference.** The share of kept groups on diff(X) is 0.39 at 0.0625, 0.38 at 0.125, 0.43 at 0.25 and 0.51 at 0.5 (0.5 = uniform). Detach leans toward the raw series at small budgets.
- **Consistency across resamples** (weighted Jaccard of per-block counts): detach 0.57 / 0.61 / 0.77 / 0.85 at 0.0625 → 0.5; random 0.56 / 0.64 / 0.73 / 0.85. **Detach's block choices are no more stable than random.** The ranking does not pick a reproducible structure.

### 4.6 Alpha

- With the authors' grid, the full model hits the grid edge (0.001) in 33% of fits, all on Beef; with the wide grid, 0%.
- Accuracy difference wide vs authors' grid: −0.03 pp (W/D/L 0/2/1). The wider grid fixes the edge hits but doesn't change accuracy here.
- Detach path steps at the wide-grid edge: 19% (Beef, at 1e-6, where the score curve is flat; see §3).

## 5. What the figures show

- **`detach_decisive_quick_accuracy_vs_speed.png`.**
  - *Left (accuracy vs measured speed-up):* `detach_stratified` (orange) sits highest at every budget except 0.0625. It is at or above full Hydra up to about 6× speed-up, then drops to 0.910 at about 8.6×. `small_g` (purple) declines steadily and is lowest at 0.0625 (0.889). Global `detach` (blue) is erratic: a dip at 0.125 (0.905), then a rise at 0.0625 (0.918). That non-monotone shape comes from Beef alone.
  - *Right (Δ vs full with 95% CIs):* every CI at every budget crosses or touches zero except `small_g`/`random`/`random_stratified` at 0.0625. The intervals are very wide (up to about ±4 pp) because n = 3.
- **`detach_decisive_quick_cd_budget_0.25.png`.** Mean ranks: full 2.67, detach_stratified 2.83, random 3.17, random_stratified 3.67, small_g 3.83, detach 4.83. One bar connects all six methods, so no difference is significant.
- **`detach_decisive_quick_cd_budget_0.125.png`.** full 1.33, detach_stratified 2.67, random_stratified 3.00, small_g 4.17, random 4.50, detach 5.33. Again a single bar over all methods. **Global detach ranks last at both budgets.** Both stratified methods rank above both global ones.
- **`detach_decisive_quick_kept_groups.png`.**
  - *Left (heatmap, detach at 0.25):* the preferred blocks differ completely between datasets. Beef keeps short-dilation raw (about 1.9× uniform) and drops long diff. SwedishLeaf keeps long dilations (about 1.8×) and drops short. Wafer keeps long raw (about 1.9×). There is no shared pattern.
  - *Right (block coverage vs budget):* global `detach` loses coverage fastest (0.81 at 0.0625), worse than global random (0.93).
- **`detach_decisive_quick_per_dataset_delta.png`.** Large bars only for Beef (+2.8, −2.8, +8.3 pp at 0.25/0.125/0.0625, changing sign between budgets). SwedishLeaf and Wafer bars are nearly invisible. This is the clearest picture of why n = 3 can't decide anything.

## 6. Verdict for this run

**Automatic verdict under the success rule: FAILURE (expected and not meaningful at n = 3).** No budget ≤ 0.25 has Holm p < 0.05. With 3 datasets the smallest possible Wilcoxon p is 0.25, so the rule *cannot* be met in QUICK_RUN.

**Early signals for the full run to confirm or reject** (hypotheses, not findings):
1. **Global `detach` ≈ `small_g`.** Its mean advantage comes entirely from Beef, it ranks last in both CD diagrams, and it is no better than global random. This matches the pilot (detach_selected ties small_g, 5/0/4).
2. **Forcing even coverage helps Detach.** `detach_stratified` beats global `detach` at the 0.5–0.125 budgets (3/0/0 at 0.125). It is also the best non-full method at 0.5–0.125 and is positive vs `small_g` at every budget (+0.6 to +2.1 pp). If this survives 40 datasets, the useful part of Detach is *ranking within blocks*, not choosing which blocks to keep.
3. **Coverage alone is not the whole story.** `random_stratified` is not consistently better than `random`, but `detach_stratified` beats `random_stratified` at every budget (+0.4 to +1.7 pp). Ranking plus coverage together is what helps.
4. **Global Detach empties blocks** (81% coverage at 0.0625), and the kept structure is no more consistent across resamples than random.

## 7. Next step: the full run

1. In the configuration cell set `QUICK_RUN = False`. Outputs then go to `detach_decisive_*` files without the `_quick` suffix, so nothing here gets overwritten or reused.
2. Run all cells, and make sure the **last cell** runs so `results/Shreed/DETACH_HYDRA_DECISIVE_FINDINGS.md` is written. Headless, from the repo root:
   ```
   cd notebooks && ../hydra_env/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 shreed_detach_hydra_decisive.ipynb > ../decisive_full.log 2>&1
   ```
3. Expected cost:
   - **Pairs:** about 200 pairs, roughly 2–4 h. In this run Wafer took about 100–110 s per pair; FordA and FordB (about 3,600 train series) will be the slowest.
   - **Timing:** plus 15–30 min of timing on 1 thread, over 6 datasets.
   - **Interruptions:** the run resumes from the last completed pair if interrupted.

**Caveats that will still apply to the full run:**
- **Splits:** the default split plus 4 stratified resamples (not 30).
- **Dataset list:** the 40 development datasets are the ones the Hydra authors also tuned on. The list comes from the team's extraction of the ROCKET appendix and couldn't be re-verified against the PDF here.
- **Timing:** on one machine and one thread.
- **Pruning step:** the elimination step is 10%.
