# Detach-Hydra Pilot: Implementation, Verification and Findings

**Project:** COMP41850, improving the HYDRA time series classifier (team: Shreed, Mahak, Aarti)
**Notebook:** `notebooks/shreed_detach_hydra.ipynb` (run end to end on 2026-11-07, 22 code cells executed in order, 0 errors)
**Outputs:**
- `results/Shreed/detach_hydra_pilot.csv`: 216 rows = 9 datasets × 3 seeds × 8 methods
- `results/Shreed/detach_hydra_pilot_paths.csv`: one row per pruning step
- `results/Shreed/detach_hydra_pilot_kept_groups.csv`: which groups survived in each selected model
- `results/Shreed/figures/detach_hydra_pruning_curves.png`
- `results/Shreed/figures/detach_hydra_speedup_vs_kernels.png`
- `results/Shreed/figures/detach_hydra_kept_groups.png`

**Runtime:** 1,572 s (about 26 min) for the pilot loop on one CPU thread. 27 of 27 dataset/seed runs succeeded and none failed.

---

## TL;DR

1. **The implementation is correct.** All six equivalence tests passed bit-for-bit. I also re-checked the results independently: three runs recomputed with separately written code match the CSVs exactly, and every internal consistency check passes.
2. **Group-level pruning works as a compression method.** Keeping **50% of groups costs −0.32 pp** of mean accuracy and gives a **1.93× faster** test transform. Keeping **25% costs −0.68 pp** and gives **3.54×**. Both are within the 1 pp target.
3. **The importance ranking is real.** At matched sizes, importance-ranked groups beat randomly chosen groups. The gap is about +2.2 pp at 5–7.5% kept, about +1 pp at 7.5–15%, about +0.4 pp at 30–60%, and about 0 above 60%. In the selected models, Detach beats random pruning on 8 of 9 datasets.
4. **The automatic selection rule is the weak point.** `detach_selected` keeps 23.5% of groups on average, but loses **−1.64 pp**, so it **fails** the 1 pp criterion. The cause is a training-CV signal that becomes misleading on small datasets. It is worst when the selection alpha is 0.001: those 7 runs prune to **6% of groups and lose 3.3 pp on average**, while their CV accuracy reads 0.96 against a test accuracy of 0.78.
5. **Against a plain smaller Hydra (`small_g`) at equal size, the result is a tie** (5/0/4 wins/draws/losses, mean −0.21 pp for Detach). The pilot therefore does **not** yet show that Detach is more than a roundabout way of choosing a smaller `g`.
6. **The speed-up tracks the kernel reduction** when 25% or more of the groups are kept (efficiency about 0.9–1.0). At very small sizes on short or small datasets it falls short, because fixed overheads dominate.
7. **A side finding: `max_only`** (dropping the hard-min features) costs −0.56 pp for a 1.52× faster transform. So the argmin and hard-count step is about a third of Hydra's transform cost.
8. **Verdict: MIXED, do not scale up as-is.** Fix the selection rule and add the `small_g` control at fixed sizes, then re-run the pilot before scaling to the 40 development datasets or all 112.

---

## 1. Motivation and idea

**Background.** Detach-ROCKET prunes ROCKET features whose ridge coefficients are small, which reduces model size with little accuracy loss. We tested whether the same idea transfers to HYDRA.

**Why per-feature pruning does not transfer.** We read the official `hydra-authors-repo/code/hydra.py`. Its forward pass works like this:

- For each dilation `d` and each input `diff_index` (0 = raw X, 1 = `torch.diff(X)`), it convolves with `W[d, diff_index]`, which holds `k·h` kernels. It then reshapes the output to `(n, h, k, T)`: `h = g/2` groups of `k = 8` kernels.
- **Soft max count:** at each time step it takes the argmax over the k kernels **of one group** and adds the winning max *values* (`scatter_add`).
- **Hard min count:** it takes the argmin over the same group and adds *ones*.
- It appends `count_max` and then `count_min` per block, concatenates along dim 1 and flattens.

Kernels compete **only inside their own group**. That has two consequences:

- **Removing a single kernel is not exact.** It changes which kernel wins at each time step in its group, so all of that group's other features change.
- **Removing a whole group is exact.** Every other feature stays bit-for-bit unchanged. That also means the group's convolutions can be deleted entirely, which gives a genuine test-time speed-up rather than just a column mask.

**Detach-Hydra:**

- **Unit of pruning:** the group `(dilation index d, diff_index, group j)`. With defaults there are `num_dilations × 2 × 32` groups (256–448 on the pilot datasets), and each owns 16 features (8 soft-max + 8 hard-min).
- **Scoring:** each group is scored by the size of its ridge coefficients.
- **Elimination:** the weakest groups are removed step by step, using training data only.
- **Rebuild:** a **smaller Hydra transform** is built that only convolves the surviving groups.

**Feature layout** (derived from the source and confirmed by Test 2):

- Blocks come in `(d, diff_index)` order, with block number `b = d·2 + diff_index`. Each block holds `2·h·k` columns, so it starts at `base = b·2hk`, and is laid out as `[max: h groups × k][min: h groups × k]`.
- Group `j` therefore owns two column ranges:
  - max columns: `base + j·k … base + (j+1)·k − 1`
  - min columns: `base + h·k + j·k … base + h·k + (j+1)·k − 1`

---

## 2. How the code was written

The notebook holds everything; no other repo file was modified. The official Hydra is imported **by file path** with `importlib`, to avoid being shadowed by the pip `hydra` package. The authors' code is not modified.

### 2.1 Building blocks
| Component | What it does |
|---|---|
| `load_ucr_dataset` | Same as Notebook 1: aeon `load_classification`, default UCR train/test split, `data/ucr_tsml`, float32, string labels |
| `group_feature_index(hydra)` | Returns `(num_dilations, divisor, h, 2k)` column indices. The flat group id is `(d·2 + v)·h + j` |
| `kept_columns(hydra, mask)` | Full-matrix columns in exactly the order `PrunedHydra` outputs them: for each block, the max columns of the kept groups, then their min columns |
| `PrunedHydra(full, keep_mask)` | Stores only the kept groups' **original** weight rows `W[d, v][j·k:(j+1)·k]`, with the same dilation and padding. Blocks with no kept group are **skipped (no convolution)**. `forward()` is a line-by-line copy of the official forward with `h` replaced by `h_kept`. It has the same `batch()` method and exposes `n_kernels` and `n_features` |
| `MaxOnlyHydra(full)` | Same convolutions, but only the soft-max features are computed (the argmin step is skipped) |
| `fit_eval` | The Notebook 1 pipeline unchanged: transform → `SparseScaler` (fitted on train only) → `RidgeClassifierCV(alphas=np.logspace(-3,3,10))`. It also measures timings and model size |

### 2.2 Sequential group detachment (`detach_path`, training data only)
1. **Fit the full model.** Full `Hydra(L, k=8, g=64, seed)`; transform; `SparseScaler` fitted on train.
2. **Choose alpha once.** `RidgeClassifierCV` on the full scaled train features sets the alpha, which is then **fixed** for the whole path.
3. **Loop:**
   - Fit `RidgeClassifier(alpha)` on the active groups.
   - **Group score** = L2 norm of `coef_` over the group's 16 columns and all classes.
   - **Selection score** = stratified k-fold CV accuracy on the training set, with `n_splits = max(2, min(5, smallest class count))`.
   - Record test accuracy and a 5-draw random-pruning curve **for plotting only**.
   - Remove the weakest 10% of active groups (at least 1). Stop at 5% of groups (at least 4).
4. **Selected model:** the smallest active set whose CV accuracy is within 0.005 of the best CV accuracy along the path.
5. **Rebuild.** `PrunedHydra(selected mask)` re-transforms train and test **from raw data**, then `SparseScaler` and `RidgeClassifierCV` are refitted. This is the deployed model. **Every rebuild asserts** that the pruned features equal the full features' kept columns.
6. **Fixed retention levels** (50%, 25%, 10%) come from the path's importance ranking: groups eliminated later rank higher, and the survivors are ordered by their final score. They are rebuilt the same way.

### 2.3 Methods compared (8 per dataset/seed)
| Method | Description |
|---|---|
| `full` | Hydra k=8, g=64 (reference) |
| `detach_selected` | Rebuilt Detach model at the CV-selected size |
| `detach_50` / `detach_25` / `detach_10` | Rebuilt Detach models at fixed retention levels |
| `random_same_size` | Same number of groups as `detach_selected`, chosen at random (5 draws; accuracy averaged, timing from draw 0). Tests whether the ranking matters |
| `small_g` | A fresh Hydra with `g' = max(2, 2·round(64·frac_kept/2))`, i.e. the same kernel budget. Tests whether pruning beats simply choosing a smaller g |
| `max_only` | Same convolutions, hard-min block dropped (half the features) |

### 2.4 Timing protocol
- `torch.set_num_threads(1)`, because the paper's speed claims are single-core.
- One warm-up call, then the median of 3 runs.
- **Latency:** (median test transform + median scale-and-predict) / n_test.
- **Model size:** Hydra weights + ridge `coef_` + `intercept_`, in bytes.

### 2.5 Pilot design
- **Datasets:** ECG200, Beef, Lightning7, CricketX, Adiac, MedicalImages, SwedishLeaf, InsectWingbeatSound, Computers. These are small to medium, not at ceiling, with lengths from 96 to 720 and 2 to 37 classes.
- **Splits and seeds:** default UCR split, seeds 0, 1 and 2.

---

## 3. Bugs encountered during development (and how they were caught)

| # | Problem | Effect | Fix |
|---|---|---|---|
| 1 | In scikit-learn 1.9.1, `RidgeClassifier.coef_` is **1-D `(n_features,)` for binary problems**, not `(1, n_features)` | `ValueError: cannot reshape array of size 4096 into shape (4096,256,16)` on ECG200 and Computers (both binary). My synthetic pre-check only used 3 classes, so it missed this | `coef = np.atleast_2d(clf.coef_)`, plus an assert on the size |
| 2 | Jupyter had the notebook open while the fix was written to disk; its next save overwrote the fix with the old in-memory code | The same error appeared again on re-run | Re-applied the fix, then reloaded the notebook from disk before running |
| 3 | `alpha_at_edge` mixed booleans with the random control's fraction-of-draws value | Read back from the CSV as strings | Stored as float |
| 4 | Hundreds of `LinAlgWarning` and "Singular matrix… using least-squares" messages | **Warnings, not errors.** Ridge with alpha = 0.001 and far fewer training series than features produces a near-singular matrix | Filtered, with a comment explaining why |
| 5 | Latency timing re-transformed the test set 4 extra times per method | Slow run (it looked hung) | Reuse the timed transform; per-stage progress prints added |

None of these affected the reported numbers in the final run: all ran on the fixed code, and the verification below confirms it.

---

## 4. Verification

### 4.1 In-notebook equivalence tests (all PASS)
| Test | Result |
|---|---|
| 1. `PrunedHydra` with all groups kept equals the official `Hydra` (ECG200 L=96, Beef L=470, train and test) | **bit-identical** (`torch.equal`) |
| 2. 5 random masks (~10–90% kept) + an "empty blocks" mask + a "diff-only" mask: `PrunedHydra(X) == Hydra(X)[:, kept_columns]` | **bit-identical in all 14 cases** |
| 3. `n_features = 2k × kept groups`, `n_kernels = k × kept groups`, weight rows, no duplicate columns | PASS |
| 4. `SparseScaler` fitted on the kept columns equals fitting on all columns and then subsetting | max abs diff = **0.0** |
| 5. ECG200 seed 0 reproduces Notebook 1 | **0.8700 = 0.8700** (alpha 215.4) |
| 6 (extra). `MaxOnlyHydra` equals `Hydra(X)[:, max_columns]` | **bit-identical** |

On top of these, the runtime rebuild assertion ran on **every** rebuilt model: 27 runs × (4 Detach models + 5 random draws), for train and test. It never failed.

### 4.2 Independent post-run verification (separate script, my own column mapping)
| Check | Result |
|---|---|
| 216 rows = 27 runs × 8 methods; kernel and feature counts consistent for every method; `small_g` g' is even | OK |
| Selected size, re-derived from the paths CSV with the rule (smallest set within 0.005 of best CV), agrees with the pilot CSV and the kept-groups CSV for all 27 runs | OK |
| Elimination schedule goes from 100% down to max(4, 5%) and is strictly decreasing | OK |
| `detach_50/25/10` keep exactly round(0.5/0.25/0.1 × groups); `random_same_size` has exactly the selected size | OK |
| **Independent recompute from raw data** (official Hydra → my own group→column mapping → SparseScaler → RidgeClassifierCV): ECG200 s0 (full 0.87, Detach 0.89), Lightning7 s0 (0.8219 / 0.7808), Beef s1 (0.8333 / 0.7667) | **all match the CSV exactly** |
| Test data used for selection? | No. Code review: the selected step, the alpha and the importance ranking use only `ytr` and the train features. Test accuracy appears only in the `*_for_plot_only` columns |

**The flagged ">2 pp gain" (ECG200 seed 0: Detach 0.89 vs full 0.87) is genuine, not a bug.** It was reproduced independently, and it amounts to 2 test series out of 100. It is noise-level, and the same dataset loses 5 pp and 4 pp on seeds 1 and 2.

---

## 5. Results

### 5.1 All methods (mean over 9 datasets × 3 seeds)
| Method | Test acc | Δ vs full (pp) | Macro-F1 | Groups kept | Features | Test-transform speed-up | Latency (ms/series) | Model size (KB) |
|---|---|---|---|---|---|---|---|---|
| full | **0.8070** | n/a | 0.7946 | 100% | 5,348 | 1.00× | 6.58 | 321.6 |
| detach_50 | 0.8037 | **−0.32** | 0.7916 | 50% | 2,674 | 1.93× | 3.50 | 160.8 |
| detach_25 | 0.8002 | **−0.68** | 0.7882 | 25% | 1,337 | 3.54× | 1.96 | 80.4 |
| detach_10 | 0.7807 | −2.63 | 0.7637 | 10% | 535 | 7.45× | 1.00 | 32.2 |
| detach_selected | 0.7905 | −1.64 | 0.7774 | 23.5% | 1,177 | 5.31× | 1.40 | 98.5 |
| random_same_size | 0.7857 | −2.13 | 0.7696 | 23.5% | 1,177 | 4.86× | 1.51 | 98.5 |
| small_g | 0.7926 | −1.44 | 0.7762 | 23.6% | 1,180 | 4.57× | 1.49 | 98.2 |
| max_only | 0.8014 | −0.56 | 0.7884 | 100% | 2,674 | 1.52× | 4.30 | 208.9 |

### 5.2 Per dataset, mean over seeds (accuracy)
| Dataset | n_train | L | full | detach_sel | random | small_g | max_only | Detach kernels kept | Detach speed-up | Δ sel vs full (pp) |
|---|---|---|---|---|---|---|---|---|---|---|
| ECG200 | 100 | 96 | 0.8833 | 0.8600 | 0.8553 | 0.8633 | 0.8800 | 8.1% | 6.45× | −2.33 |
| Beef | 30 | 470 | 0.8333 | 0.8111 | 0.8089 | 0.8444 | 0.8444 | 7.3% | 4.55× | −2.22 |
| Lightning7 | 70 | 319 | 0.8265 | 0.7763 | 0.7607 | 0.7717 | 0.7991 | 5.9% | 10.36× | −5.02 |
| CricketX | 390 | 300 | 0.7906 | 0.7846 | 0.7638 | 0.7761 | 0.8060 | 26.9% | 3.69× | −0.60 |
| Adiac | 390 | 176 | 0.8201 | 0.8082 | 0.7986 | 0.7980 | 0.7971 | 49.7% | 2.06× | −1.19 |
| MedicalImages | 381 | 99 | 0.7667 | 0.7623 | 0.7524 | 0.7526 | 0.7570 | 35.6% | 3.08× | −0.44 |
| SwedishLeaf | 500 | 128 | 0.9808 | 0.9771 | 0.9727 | 0.9755 | 0.9712 | 44.5% | 2.28× | −0.37 |
| InsectWingbeatSound | 220 | 256 | 0.6508 | 0.6500 | 0.6468 | 0.6520 | 0.6364 | 21.5% | 6.14× | −0.08 |
| Computers | 250 | 720 | 0.7107 | 0.6853 | 0.7123 | 0.7000 | 0.7213 | 12.2% | 9.22× | −2.53 |
| **Mean** | | | **0.8070** | **0.7905** | **0.7857** | **0.7926** | **0.8014** | **23.5%** | **5.31×** | **−1.64** |

### 5.3 Accuracy difference vs full Hydra, per dataset (pp, mean over seeds)
| Dataset | detach_50 | detach_25 | detach_10 | detach_selected | random_same_size | small_g | max_only |
|---|---|---|---|---|---|---|---|
| Adiac | +0.09 | −1.88 | −6.31 | −1.19 | −2.15 | −2.22 | −2.30 |
| Beef | 0.00 | +1.11 | −4.44 | −2.22 | −2.44 | +1.11 | +1.11 |
| Computers | −1.20 | −1.73 | −4.27 | −2.53 | +0.16 | −1.07 | +1.07 |
| CricketX | −0.17 | −1.37 | −0.85 | −0.60 | −2.68 | −1.45 | +1.54 |
| ECG200 | −0.67 | 0.00 | −0.33 | −2.33 | −2.80 | −2.00 | −0.33 |
| InsectWingbeatSound | +0.44 | −0.03 | +0.22 | −0.08 | −0.40 | +0.12 | −1.45 |
| Lightning7 | 0.00 | −1.37 | −3.20 | −5.02 | −6.58 | −5.48 | −2.74 |
| MedicalImages | −0.92 | −0.22 | −2.59 | −0.44 | −1.43 | −1.40 | −0.96 |
| SwedishLeaf | −0.48 | −0.59 | −1.92 | −0.37 | −0.81 | −0.53 | −0.96 |
| **Mean** | **−0.32** | **−0.68** | **−2.63** | **−1.64** | **−2.13** | **−1.44** | **−0.56** |

### 5.4 Win/draw/loss of `detach_selected` (descriptive only, no significance claims; 9 datasets × 3 seeds is a pilot)
| detach_selected vs | Datasets W/D/L | Runs W/D/L | Mean Δ (pp) |
|---|---|---|---|
| full | 0/0/9 | 5/3/19 | −1.64 |
| random_same_size | **8/0/1** | 19/0/8 | +0.48 |
| small_g | 5/0/4 | 15/2/10 | −0.21 |
| max_only | 4/0/5 | 11/4/12 | −1.09 |

### 5.5 Ridge alpha (the selection alpha, chosen by `RidgeClassifierCV` on the full features)
- **Chosen values:** 215.4 in 16 runs, **0.001 in 7 runs**, 1000 in 4 runs.
- **At a grid edge:** 11 of 27 runs.
- **Bimodal:** the choice jumps between the bottom edge and the top of the grid, even across seeds of the same dataset (ECG200: 215, 0.001, 215).
- **Consistent with our earlier finding.** This repeats the team's earlier observation that `logspace(-3, 3, 10)` is too coarse and too narrow for Hydra.

---

## 6. Findings from the figures

### 6.1 `detach_hydra_pruning_curves.png` (accuracy vs fraction of groups kept, log x-axis, mean over seeds)
- **Importance beats random almost everywhere, and most clearly when pruning is aggressive.**
  - On **Adiac, MedicalImages, CricketX, SwedishLeaf and InsectWingbeatSound**, the blue Detach curve lies clearly above the orange random curve over the whole range below about 50%. On Adiac at 10% kept it is about 0.755 vs 0.73, and on CricketX at 5% about 0.765 vs 0.72.
  - Averaged over all paths (plot-only, fixed alpha, matched sizes), **Detach minus random** is +2.15 pp at 5–7.5% kept, +1.05 pp at 7.5–15%, +0.84 pp at 15–30%, +0.37 pp at 30–60% and −0.04 pp at 60–100%.
- **The curves are flat down to about 25–50%.** CricketX, Beef, Lightning7 (≥25%), ECG200 and Computers stay within about 1 pp of the full-Hydra line until roughly 20–30% of groups remain. Most of Hydra's groups are redundant for the linear classifier on these datasets.
- **Pruning can act as mild regularisation.** On **InsectWingbeatSound**, Detach rises *above* full Hydra (about 0.66 vs 0.651) between 12% and 25% kept. Its selection alpha was at the **top edge (1000)** in all 3 seeds. Both point to the model wanting more regularisation than the grid allows.
- **No signal on Beef and Computers.** Detach and random overlap, and both wander around the full line. Beef has 30 test series (one series = 3.3 pp). Computers' classes appear to depend on features spread across many groups.
- **The grey dotted CV curve explains the selection failures:**
  - On **ECG200, Beef, Lightning7 and Computers**, training-CV accuracy sits far above test accuracy (0.86–0.98 vs 0.70–0.88). It also **rises** as groups are removed: Beef goes from 0.71 at 100% to 0.93 at 5%, and Lightning7 from 0.73 to 0.91. A rule that picks "the smallest set within 0.005 of best CV" therefore picks nearly the smallest set, while test accuracy is falling.
  - On **Adiac, MedicalImages and SwedishLeaf**, CV follows the shape of the test curve, and the selection lands at a sensible 36–50% kept.
- **The rebuilt Detach models sit on or near the plot-only curve.** That confirms the rebuilt transform plus re-selected alpha behaves like the column-subset path.

### 6.2 `detach_hydra_speedup_vs_kernels.png` (speed-up vs fraction of kernels removed; dashed = ideal 1/fraction kept)
- **At 50% and 25% kept, the points sit on the ideal curve.** Measured as speed-up × fraction kept (1.0 = proportional), the median is 0.99 at 50% and 0.93 at 25%. The test transform's cost is dominated by the convolution and argmax work, which scales with the number of kernels.
- **At 10% kept, the points fall below the ideal**: median efficiency 0.80, with speed-ups spread from about 3.6× to 10×. The worst are the short or small datasets (Beef 0.38, ECG200 0.56, Lightning7 0.67). For these the absolute time is tiny (Beef's full transform takes 0.36 s), so fixed costs dominate: the `torch.diff`, the per-block Python loop and tensor allocation.
- **Large datasets stay proportional even at 10%** (Computers 0.92, InsectWingbeatSound 0.93).
- **The `detach_selected` points follow the same curve.** The largest observed speed-up is about 15× (Computers seed 0, 6% kept).
- **Detach is slightly faster than `small_g` at equal kernel count** (5.31× vs 4.57× mean). That is partly because empty `(d, diff)` blocks are skipped entirely.

### 6.3 `detach_hydra_kept_groups.png` (which groups survive in `detach_selected`)
- **Dilation (heatmap of fraction kept per dilation):**
  - **SwedishLeaf** strongly prefers dilation 2³, keeping 71% of those groups vs 27–31% at 2⁰–2¹. **MedicalImages** keeps more at higher dilations: 15% at 2⁰ rising to 55% at 2³. **InsectWingbeatSound** keeps 50% at 2⁴ vs about 10% at 2⁰–2¹. **CricketX** gradually prefers larger dilations (20% → 37% at 2⁴).
  - So for shape-like and long-range datasets, **long-dilation groups carry more signal**, and short-dilation groups are pruned first.
  - **Adiac** is flat at about 50% across dilations. **Beef, Lightning7, ECG200 and Computers** are uniformly dark because almost everything was pruned there (the over-pruning problem).
- **Raw X vs first difference (bar chart; 50% = no preference):**
  - **Computers keeps almost only raw-X groups (4.5% diff).** **Adiac prefers diff(X) (74%).** ECG200 (27%), Lightning7 (29%) and MedicalImages (30%) lean towards raw X. Beef, CricketX, SwedishLeaf and InsectWingbeatSound are near 50%.
  - **Interpretation:** which input representation matters depends strongly on the dataset. A fixed 50/50 split of groups between X and diff(X), as in Hydra, is not optimal for every dataset. This could motivate a learned or CV-chosen ratio.

---

## 7. Diagnosis: why `detach_selected` underperforms the fixed 25–50% levels

**1. With a fixed alpha of 0.001, coefficient norms are unreliable as importance scores.** Grouped by selection alpha:

| Selection alpha | Runs | Mean groups kept | Mean Δ vs full (pp) | Mean CV acc (train) | Mean test acc |
|---|---|---|---|---|---|
| 0.001 (bottom edge) | 7 | **6.2%** | **−3.30** | 0.961 | 0.784 |
| 215.4 | 16 | 30.2% | −1.35 | 0.895 | 0.821 |
| 1000 (top edge) | 4 | 27.1% | **+0.07** | 0.690 | 0.679 |

- With alpha = 0.001 and n_train ≪ n_features, the ridge nearly interpolates the training data. CV at 100% of groups is poor (Beef 0.71). As groups are removed, the fixed tiny alpha becomes effectively better regularised, so CV *rises*, and the path always ends near the 5% floor.
- The 7 runs at alpha 0.001 include all 3 Beef seeds, Lightning7 seeds 0 and 2, ECG200 seed 1 and Computers seed 0. Most of them are among the worst losses.

**2. The CV signal is noisy on small training sets.**
- Beef has 30 training series and 5 classes (6 per class, folds of 6), so one CV error moves the score by 3.3 pp. A 0.005 tolerance is therefore effectively "exact ties only", and the rule then picks the smallest of the tied sets.
- The mean CV accuracy of the selected model is 0.882, against a mean test accuracy of 0.791. The selection is clearly optimistic.

**3. Selection bias and a small leak.**
- Choosing the best of about 31–34 path steps on the same CV folds inflates the selected CV score.
- `SparseScaler` statistics inside the CV come from the full training set (accepted for the pilot).

**4. Where selection works.** On the datasets with 220–500 training series (CricketX, Adiac, MedicalImages, SwedishLeaf, InsectWingbeatSound), `detach_selected` loses only −0.08 to −1.19 pp, while keeping 21–50% of groups for a 2–6× speed-up.

---

## 8. Go / no-go verdict

| Question | Answer | Evidence |
|---|---|---|
| Q1. Does `detach_selected` keep ≤ 50% of groups while staying within 1 pp of full Hydra? | **NO** for the automatic rule; **YES** for the fixed levels | Selected: 23.5% kept, −1.64 pp. Fixed: `detach_50` −0.32 pp (1.93×), `detach_25` −0.68 pp (3.54×) |
| Q2. Does it beat `random_same_size` and `small_g` at equal size on most datasets? | **Random: YES. small_g: INCONCLUSIVE** | vs random: 8/9 datasets, +0.48 pp, and +0.4 to +2.2 pp on the paired curves. vs small_g: 5/0/4, mean −0.21 pp |
| Q3. Is the test-transform speed-up roughly proportional to the kernel reduction? | **YES at ≥ 25% kept; partly at 10%** | Efficiency median 0.99 at 50%, 0.93 at 25%, 0.80 at 10%; worse for short or small datasets because of fixed overheads |

**Recommendation: MIXED. Do not scale up the current version yet.** The mechanism is correct and the ranking carries real signal. But the automatic selection rule over-prunes on small datasets. And the decisive comparison, Detach vs a smaller Hydra at the *same* size, is so far a tie, which makes Detach look like a hyperparameter change. Before scaling to the 40 development datasets or all 112:

1. **Fix the selection rule:**
   - Re-select alpha at every step, or at least use a wider and finer grid such as `logspace(-3, 5, 30)`, and never rank with alpha at the bottom edge.
   - Use a 1-standard-error rule, or a tolerance scaled to the CV granularity (for example 1/n_train), instead of a fixed 0.005.
   - Use repeated stratified CV to reduce noise.
   - Optionally set a minimum retention floor (for example 25%), since the fixed levels show 25–50% is safe.
2. **Add `small_g` at the fixed levels.** That means g' = 32, 16 and 6 against `detach_50`, `detach_25` and `detach_10`, together with `random` at those levels with a real rebuild. This is the test that decides whether the ranking beats a smaller random Hydra of the same size. The plot-only curves already beat random by +0.4 to +2 pp at 15–60%, which is encouraging.
3. **Remove the scaler leak** by fitting `SparseScaler` inside each CV fold.
4. **Report wall-clock cost.** The pilot loop took about 26 min for 27 runs on one thread. The path itself is cheap (1–3 s); most of the time goes on timing the 8 methods with repeats.

**Worth following up regardless:**
- **`max_only`** gives a 1.52× speed-up for −0.56 pp, and it even wins on CricketX, Computers and Beef. Combining max-only with group pruning could stack the two speed-ups.
- **The kept-group analysis** suggests dataset-specific dilation and raw-vs-diff preferences. A CV-chosen X:diff(X) group ratio is a cheap idea to test.

---

## 9. Caveats
- **Pilot scale.** 9 datasets × 3 seeds on the single default UCR split (not the paper's 30 resamples), so there are **no significance claims**. Small test sets (Beef 30, ECG200 100, Lightning7 73) make single-dataset differences of a few pp noise-level.
- **Seeds.** They change Hydra's random kernels, the CV folds and the random draws, but not the train/test split.
- **Alpha.** It is fixed during the elimination path. The rebuilt models re-select it with `RidgeClassifierCV`.
- **Timings.** Single-thread CPU on one Mac (median of 3 after warm-up). Absolute numbers are machine-specific; the ratios are what matter.
- **Plot-only quantities.** The random curve in the pruning plot uses the fixed path alpha and column subsets, not rebuilt models. It is a like-for-like comparison with the Detach path curve (same alpha), but it is **not** used for any decision.
- **Verdict cell.** The notebook's verdict markdown cell is still the template. This document is the filled-in verdict.
