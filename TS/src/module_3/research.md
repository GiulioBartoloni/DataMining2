# Module 3 — Time Series Classification (sii): Research Notes

> Purpose: a self-contained briefing for planning Module 3 of the DM2 project.
> Scope is **only Time Series Classification of the binary `sii` target**. It
> consolidates (a) the exact structure of the preprocessed dataset, (b) the
> course methods that apply, (c) the project's existing evaluation conventions,
> and (d) the tooling situation. Feed this to the planner.

---

## 1. Task definition (from the guidelines)

Module 3 — *Time Series Classification* requires us to **define one (or more)
classification task and solve it using**:

- **KNN with at least two distances**: (1) Euclidean **or** Manhattan, and (2) **DTW**.
- **Shapelets**: analyse the shapelets retrieved.
- **At least one other method** (ROCKET, MUSE, CNN, RNN, etc.).
- *(Optional)* **Sequential Pattern Mining**: discretize the series and mine frequent patterns/trends.

The target is the child's **Severity Impairment Index (sii)**. In the time-series
dataset the sii has been **binarized**: `sii = 0` (non-problematic) vs `sii ∈ {1,2,3}`
(problematic) → a single binary column `sii_binary`. **So Module 3 is a binary
classification problem**, not the 4-class problem of the tabular Module 2.

> Note: Module 0 (preprocessing of the time-series dataset) is already done; this
> module consumes its output. Motifs/discords and clustering are *separate*
> Module 3 sub-tasks and are **out of scope** here (user scoped this to classification only).

---

## 2. The preprocessed dataset — exact structure

**File:** `TS/processed_data/ts_preprocessed.pkl.gz` (also mirrored at
`TS/src/processed_data/ts_preprocessed.pkl.gz`). Gzipped pickle of a **Python list
of pandas DataFrames**, one DataFrame per time series.

```python
import gzip, pickle
with gzip.open("TS/processed_data/ts_preprocessed.pkl.gz", "rb") as f:
    data = pickle.load(f)        # list of DataFrames
```

| Property | Value |
|---|---|
| Number of series | **4 252** |
| Length of every series | **200 timesteps** (all equal) |
| Columns per series | `['X', 'Y', 'Z', 'enmo', 'anglez', 'id', 'sii_binary']` |
| Signal channels | **5**: `X, Y, Z, enmo, anglez` (`float32`) |
| Metadata columns | `id` (`int64`), `sii_binary` (`int64`) — constant within a series |
| Target | `sii_binary` ∈ {0, 1} |

So the modelling object is a **multivariate time series**: shape **(4252 series ×
200 timesteps × 5 channels)** if stacked into a 3-D array. The `id` and
`sii_binary` columns are repeated down all 200 rows of each DataFrame (one label
per series).

### Channel semantics & value ranges (over all 4252×200 points)

| Channel | Meaning | min | max | mean | std |
|---|---|---|---|---|---|
| `X` | accelerometer x (g) | −1.466 | 1.361 | −0.097 | 0.545 |
| `Y` | accelerometer y (g) | −2.084 | 1.552 | 0.002 | 0.354 |
| `Z` | accelerometer z (g) | −1.042 | 1.183 | −0.090 | 0.400 |
| `enmo` | Euclidean Norm Minus One (g), clipped ≥0 — activity intensity | 0.000 | 2.366 | 0.059 | 0.084 |
| `anglez` | arm-elevation angle (degrees) | −89.86 | 89.45 | −6.45 | 27.65 |

**Scales differ by ~2 orders of magnitude** (`enmo` ≈ [0, 2.4] vs `anglez` ≈
[−90, 90]). Any distance-based or scale-sensitive method needs per-channel
scaling — see normalization caveat (§4).

### Class distribution

| Level | Series-level | Subject-level (326 ids) |
|---|---|---|
| class 0 (non-problematic) | 2 852 (**67.1 %**) | 206 (63.2 %) |
| class 1 (problematic) | 1 400 (**32.9 %**) | 120 (36.8 %) |

Mild imbalance (~2:1). Far less extreme than the tabular task — no need for the
heavy resampling machinery of Module 1, but **macro / balanced metrics and
`class_weight='balanced'` are still warranted**.

---

## 3. ⚠️ Critical structural fact: many series per subject (group leakage)

- **4 252 series come from only 326 unique subjects (`id`).**
- Series per subject: **min 1, max 36, mean 13, median 11.** (81 subjects have a
  single series; the rest have many — one subject contributes 36 series.) The
  multiple series are different **days** of the same child (one series = one
  subject-day, see §9).
- `sii_binary` is **100 % consistent within a subject** (0 subjects have >1 distinct
  label). The label is a **subject-level attribute**.
- **Data is concentrated in a few prolific subjects:** the top 10 % of subjects
  hold ~24 % of all series; the top 19 % hold ~44 %.
- **#days weakly anti-correlates with the label.** Class-1 (problematic) children
  contribute fewer days (mean 11.7, median 9) than class-0 (mean 13.8, median 13);
  point-biserial corr(label, n_series) = −0.097 (p ≈ 0.08, borderline). This is
  why **prevalence shifts between granularities: 32.9 % problematic at series level
  vs 36.8 % at subject level.**

### 3.1 The split (the headline consequence)

A naive random/stratified split over the 4 252 series puts different days of the
*same child* in both train and test. Days of one child share the same label and a
very similar activity signature, so that is **label leakage** and inflates every
metric. **The split must be subject-grouped** (`StratifiedGroupKFold` /
`GroupShuffleSplit` on `id`, stratified by label), and grouping must hold at
**every CV level** — the inner `RandomizedSearchCV`/threshold-tuning loop also
needs `StratifiedGroupKFold`, else the same child's days leak across inner folds
and the model is over-tuned. Use **nested grouped CV** throughout. Report the
grouped result as the headline; optionally show the naive-split number to quantify
the leakage gap.

### 3.2 It changes the methodology, not just the split

1. **Effective sample size is ≈ 326, not 4 252.** The series are ~13 correlated
   replicates of 326 independent units, so the statistical evidence for
   *generalizing to new children* is ~326 label draws. ⇒ a single split is noisy;
   report **mean ± std across grouped folds**, and don't trust the tight
   confidence implied by "4 252 samples".
2. **KNN is the most exposed method.** Another day of the *same* child is almost
   always a closer match than any other child (same gait, wrist dominance, device
   fit, baseline activity). Grouping removes the worst form (test child absent from
   train) — which is *why* grouping is non-negotiable for KNN specifically — but
   even then the neighbour pool is dominated by prolific subjects (top 19 % of
   subjects = 44 % of series), biasing the vote toward a few children.
3. **Shapelet / feature / info-gain models inherit the same bias.** Shapelet
   quality is scored by information gain **over training series, each counted
   equally**; a subsequence can score high simply because it is a quirk of one
   36-day subject whose label it matches — a **subject fingerprint masquerading as
   a class shapelet**. Same risk for any feature-selection step (catch22, MUSE word
   selection).
4. **High-capacity models (ROCKET ~20 k features, deep nets) can fingerprint the
   subject** instead of the condition. Grouped evaluation keeps the test metric
   honest, but capacity is wasted and generalization suffers — an argument for the
   strong regularization the tabular report already favoured.
5. **Mitigations for the prolific-subject dominance** (effects 2–4): weight each
   series by `1/n_days(id)`, and/or **cap days per subject** (e.g. ≤ k random days
   each) — which doubles as the principled way to subsample for expensive DTW
   (§4.3), since it preserves *subject diversity* rather than dropping at random.

### 3.3 The honest unit of evaluation is the subject

Because `sii` is a subject attribute and series-level prevalence is distorted
(32.9 % vs 36.8 %), the methodologically clean target is a **per-subject
prediction**: classify each day, then **pool a child's days (mean probability or
majority vote) and evaluate on the 326 subjects**. Plan: report **subject-level
metrics as the headline** and series-level for completeness. (Per-series training
with grouped CV is still the natural default — it gives more training data — but
the *evaluation* unit should be the subject.)

---

## 4. Data caveats & decisions the plan must address

1. **Subject-grouped, stratified splitting + nested grouped CV** (see §3.1) —
   mandatory; **subject-level evaluation** as the headline metric (see §3.3);
   consider `1/n_days` series weighting or day-capping to neutralize prolific-
   subject dominance (see §3.2).
2. **Normalization choice is non-trivial.** The course note warns: remove a
   distortion (offset / amplitude / trend) *only if it is noise for the task,
   keep it if it carries signal*.
   - For accelerometer data the **per-axis mean of X/Y/Z encodes limb orientation
     (gravity direction)** and the **level of `enmo` encodes activity intensity** —
     these are physiologically meaningful, so blind per-series z-normalization may
     destroy signal.
   - But cross-channel scale differences (enmo vs anglez) must be handled for
     distance methods. Reasonable default: **per-channel scaling computed on the
     training set** (z-score or min-max per channel, fit on train only), rather
     than per-series normalization. Flag this as an explicit, justified choice.
3. **Computational scale.** Pairwise DTW over 4 252 series of length 200 across 5
   channels is heavy (KNN-DTW is O(N²·m²) at predict-with-LOO, ~18M pairs ×
   200² ops). Mitigations from the notes: a **Sakoe-Chiba band** (warping window
   2–8 % of length is usually optimal), **PAA/SAX downsampling before DTW** (matrix
   shrinks quadratically with length), and/or **working on a subsample** of series.
   The guidelines explicitly bless approximation (SAX/PAA) when the data is too
   large. Plan should budget for this.
4. **Multivariate handling.** 5 channels. Two strategies from the note:
   (a) treat channels independently and **concatenate** into one long univariate
   series (works with any univariate method), or (b) use **natively multivariate**
   methods (multivariate KNN-DTW with dependent/independent DTW, ROCKET family,
   MUSE, InceptionTime). Decide per method; consider also a **channel-subset
   study** (e.g. `enmo`+`anglez` are the classic actigraphy sleep/activity signals).
5. **Series length vs "short/long" guidance.** 200 points is *short-to-medium* →
   the note recommends **shape-based distances (DTW)** for short series, and
   structural/feature-based for long series. Both are in-scope; DTW is the
   expected primary distance.

---

## 5. Methods — distilled from the course notes (Notes 15 & 16)

### 5.1 Distances (Note 15)

- **Euclidean** (rigid one-to-one alignment): `D(Q,C)=√Σ(qᵢ−cᵢ)²`. Fast but
  cannot tolerate temporal shift/acceleration; very sensitive to offset/
  amplitude/trend/noise → normalize first.
- **Manhattan** = the diagonal-only DTW path with `d=|x−y|`.
- **DTW (Dynamic Time Warping)** — the elastic distance that the project requires
  as the 2nd KNN distance:
  - Non-linear warped alignment via DP recurrence
    `γ(i,j)=d(qᵢ,cⱼ)+min{γ(i−1,j−1), γ(i−1,j), γ(i,j−1)}`; distance = `γ(n,m)`.
  - **Always ≤ Manhattan** (diagonal path is one allowed path); a warping band can
    only *increase* the result.
  - **Global constraints** to speed up & regularize: **Sakoe-Chiba band** (constant
    width `r` around diagonal) or **Itakura parallelogram**. Empirically **best
    accuracy at a small window, ~2–8 % of length**; wider can hurt. ⇒ treat the
    band width as a tunable hyperparameter.
  - Speed-up: run DTW on a **PAA/SAX-approximated** (downsampled) series.
  - **1-NN-DTW is the classic strong baseline** every TSC method is compared to.

### 5.2 Instance-based: k-NN (Note 16, the project's KNN requirement)

To label a test series: compute distance to all training series, take the `k`
nearest, majority-vote. Stores the whole training set; slow at predict time. The
distance is the crucial ingredient. Deliverable: **KNN with Euclidean/Manhattan
AND with DTW**, compared (the note's own comparison method is 1-NN + LOO accuracy).
Tunables: `k`, distance, DTW band width, normalization.

### 5.3 Shapelet-based (Note 16, the project's Shapelet requirement)

- A **shapelet** = a subsequence maximally discriminative for a class.
  `subsequenceDist(T,S)=min over S' in T of Dist(S,S')` (distance of S to its
  best-matching location in T).
- **Pipeline (Shapelet Transform):** (1) extract `K` discriminative shapelets from
  train; (2) transform each series → vector of its `K` distances to the shapelets;
  (3) train any standard classifier (DT, KNN, logistic, etc.) on the `N×K` table.
  **Interpretable** — each feature is a distance to a concrete, plottable shape.
- Shapelet quality is scored like a DT split: order series by distance to the
  candidate, pick the split maximizing **information gain**
  `Gain = I(D) − [f(D₁)I(D₁)+f(D₂)I(D₂)]`.
- Extraction methods: **brute force** (with *Distance Early Abandon* + *Admissible
  Entropy Pruning* — needed because #candidates explodes), **random**,
  **gradient-based** (learn shapelets by SGD), and SAX/SFA sequence-learner
  variants (**MrSEQL, MrSQM**). A modern fast variant is **RDST (Random Dilated
  Shapelet Transform)**.
- **Deliverable:** retrieve shapelets, **plot & interpret the top discriminative
  shapes** (what activity pattern separates problematic vs non-problematic), and
  classify on the shapelet-transformed features.

### 5.4 "At least one other method" — candidates (Note 16)

Pick **≥1** (recommend the kernel family for accuracy/speed, optionally add a deep net):

- **Kernel-based — ROCKET family (recommended primary "other method").**
  ROCKET convolves each series with ~10 000 random kernels, pools each activation
  map by **max + PPV** (~20 000 features), then a **ridge/logistic linear
  classifier**. Accurate, fast, scalable; only hyperparameter is #kernels.
  - **MINIROCKET** (fixed kernels, PPV only, ~10k features, up to 75× faster),
    **MultiROCKET** (adds first-differences + 3 extra poolings, ~50k features),
    **Hydra**/**MultiROCKET-Hydra** (kernel+dictionary hybrid). **Arsenal** = ROCKET
    ensemble. All handle multivariate natively.
- **Dictionary-based — MUSE (WEASEL+MUSE)** — the project lists MUSE explicitly.
  Native multivariate bag-of-SFA-symbols; builds per-channel words tagged by
  sensor+window, classified with logistic regression. (BOSS/WEASEL are the
  univariate ancestors.)
- **Deep learning — CNN / InceptionTime / LSTM-FCN / ResNet.** 1-D convolutions on
  the raw series; **InceptionTime** (ensemble of Inception nets) and
  **MultivariateLSTM-FCN** are the strong TS-specific architectures. Note: these
  need a DL backend (TensorFlow/PyTorch) **which is not installed** (§7).
- **Interval-based — TSF / CIF / DrCIF**, and **feature-based — catch22 / TSFresh**
  + a standard classifier (a cheap, interpretable extra baseline; Note 15).
- **Hybrid — HIVE-COTE 2.0 / TS-CHIEF** are SOTA but very slow — mention as the
  literature ceiling, not a required run.

> Benchmark takeaway (Note 16): hybrids (HIVE-COTE 2.0) and recent kernels
> (MultiROCKET-Hydra) top accuracy; **ROCKET hits near-top accuracy at a fraction
> of the time**; older baselines (1-NN-DTW, plain shapelet/interval, CNN) rank
> lower. Good story for the report: cheap baseline (KNN) → interpretable
> (shapelets) → strong/fast (ROCKET).

### 5.5 Approximation & feature representations (Notes 15–16) — supporting tools

- **PAA** (mean of equal segments) and **SAX** (PAA → symbols over an
  equiprobable-Gaussian alphabet) — time-dependent, instance-wise. Use to
  **speed up DTW** and to **discretize for sequential pattern mining**. SAX
  `MINDIST` lower-bounds Euclidean.
- **SFA** (DFT → data-adaptive MCB binning) underlies BOSS/WEASEL/MUSE.
- **Ordering rule:** DTW is meaningful after a *time-dependent* approximation
  (PAA, SAX) but **not** after a time-independent one (DFT, SFA, SVD, PCA).
- **Structural / global features** (catch22, TSFresh, simple stats) → fixed-length
  vector → ordinary classifier. Good for a feature-based baseline & XAI.

### 5.6 Optional — Sequential Pattern Mining

Discretize each (per-channel) series with **SAX**, then run frequent-sequence /
sequential-pattern mining to surface recurring symbolic motifs/trends that
correlate with the label. Optional deliverable; ties to the dictionary view.

---

## 6. Evaluation protocol (mirror the project's existing conventions)

From the existing report (Module 2 classification), reuse the same rigor, adapted
to **binary** classification and **grouped** splitting:

- **Split:** subject-grouped + stratified train/test (e.g. 80/20 via
  `StratifiedGroupKFold`/`GroupShuffleSplit` on `id`), `random_state=42`. *(The
  existing tabular work used a plain stratified 80/20; here grouping is mandatory —
  see §3.1.)*
- **Nested grouped CV:** the tuning loop (`RandomizedSearchCV`/`GridSearchCV`) must
  *also* use `StratifiedGroupKFold` (e.g. 5-fold) — grouping at every level, or the
  inner folds leak. Score **F1** (binary) or **balanced accuracy / macro-F1**.
  Always **justify the grid** (which params, why, which won) — the guidelines demand it.
- **Evaluation unit — subject-level is the headline (§3.3):** pool each child's
  per-day predictions (mean probability or majority vote) and score on the **326
  subjects**; report series-level for completeness. Because the effective sample
  size is ≈ 326 (§3.2), report **mean ± std across grouped folds**, not a single split.
- **Imbalance:** `class_weight='balanced'` where supported (mild 67/33 series,
  63/37 subject). Optionally `1/n_days` series weighting to curb prolific-subject
  dominance (§3.2). Heavy resampling is unnecessary here.
- **Metrics:** accuracy, precision, recall, **F1**, plus **ROC curve / ROC-AUC**
  (and/or precision-recall curve), **confusion matrix**. Report per-class.
- **Per-method discussion** (guidelines): for KNN — effect of `k` and of distance
  (Euclidean vs DTW) and band width, *and the subject-neighbour bias* (§3.2);
  for shapelets — *what shapes were retrieved and what they mean*, guarding against
  subject-fingerprint shapelets; for ROCKET/deep — accuracy vs training-time
  trade-off, overfitting/fingerprinting controls. Cross-method comparison table
  (like `tab:cmi_clf`).
- **Baselines:** a majority-class / `DummyClassifier` reference (series-level
  accuracy ≈ 0.671; subject-level ≈ 0.632), plus optionally the *naive (ungrouped)
  split* number to demonstrate the leakage gap.
- **XAI hook (optional, ties to Module 2):** shapelet-based explanations are
  intrinsically interpretable; SHAP/saliency can explain ROCKET/deep models. Note
  16 has a full XAI-for-TSA taxonomy if we want an explainability paragraph.

---

## 7. Tooling — current state & what the plan must add

**No time-series classification library is installed.** The venv
(`.venv`, Python ≥ 3.14 per `pyproject.toml`) has: `scikit-learn 1.8`,
`numpy 2.4`, `pandas 3.0`, `scipy 1.17`, `xgboost 3.2`, `shap`, `imblearn`,
`pyod`, `matplotlib`, `seaborn`. **Missing:** `aeon`, `sktime`, `tslearn`,
`pyts`, `stumpy`, `torch`, `tensorflow`.

**Recommendation:** add **`aeon`** (modern, actively maintained successor to
sktime) — it covers *every* required method under one API:

| Requirement | aeon estimator |
|---|---|
| KNN + Euclidean/DTW | `KNeighborsTimeSeriesClassifier(distance="euclidean" / "dtw")` (multivariate-capable) |
| Shapelets | `ShapeletTransformClassifier`, `RDSTClassifier`, `MrSQMClassifier` |
| ROCKET family | `RocketClassifier`, `MiniRocket`, `MultiRocket`, `Arsenal` |
| Dictionary (MUSE) | `MUSE`, `WEASEL`, `BOSSEnsemble` |
| Deep (needs backend) | `InceptionTimeClassifier`, `CNNClassifier`, `LSTMFCNClassifier` |
| Feature-based | `Catch22Classifier`, `TSFreshClassifier` |
| Approximation | PAA/SAX/SFA transformers |

Alternatives: `tslearn` (KNN-DTW, shapelets, good DTW utilities), `pyts`
(ROCKET, BOSS/SAX-VSM, imaging), `sktime` (same families, heavier), `stumpy`
(matrix profile — only if motifs are added later).

> ⚠️ **Compatibility risk to verify in the plan:** `pyproject.toml` pins
> `requires-python = ">=3.14"`. `aeon`/`sktime`/`tslearn` and especially deep-
> learning backends (TF/PyTorch) **may not yet publish Python 3.14 wheels**. The
> plan should (a) verify install on 3.14, and (b) if the DL "other method" can't
> install, fall back to the **ROCKET family or MUSE** (no DL backend needed) to
> satisfy the "≥1 other method" requirement. Numpy 2.x / pandas 3.x are also new
> enough to occasionally break older TS libs — verify early.

**Data-shape adapter:** aeon/sktime expect a 3-D array `(n_cases, n_channels,
n_timepoints)` or a nested/`pd-multiindex` frame. The list-of-DataFrames must be
converted: stack the 5 channel columns of each series into shape
**`(4252, 5, 200)`**, with a separate `y` vector of `sii_binary` (length 4252) and
a `groups` vector of `id` (length 4252) for grouped CV.

---

## 8. Current repository state

```
TS/
├── base_data/cmi_timeseries_dataset_dm2_25_26.zip   # raw (4437 series; pkl.gz + instruction.txt inside)
├── processed_data/ts_preprocessed.pkl.gz            # ← MODULE 3 INPUT (4252 series)
└── src/
    ├── module_0/preprocess.ipynb                    # done: load→missing→colselect→non-wear→spike→export
    ├── module_3/                                     # EMPTY (this is where the work goes; research.md lives here)
    └── processed_data/ts_preprocessed.pkl.gz         # mirror copy
report_dm2.tex                                         # report; has NO time-series section yet (modules 0–2 tabular only)
```

**Preprocessing funnel (Module 0, for context in the report):**
`4437 raw → 4258 fully-worn (dropped 179 with any non-wear) → 4252 clean (dropped 6
saturation/clipping artifacts)`; total **4.2 % dropped**. Unique subjects fell
**494 → 326** (subjects whose every day had non-wear were removed entirely).
Channels kept: `X,Y,Z,enmo,anglez`; dropped: `non-wear_flag` (now all 0),
`light`, `battery_voltage`, and constant-per-series metadata
`weekday, quarter, relative_date_PCIAT`.

**Report conventions to match:** `\imgph{...}` placeholders for figures; per-model
"Model and tuning" + "Results" paragraphs; a consolidated comparison table; macro/
balanced metrics emphasized; `random_state=42`; explicit leakage discussion (the
tabular report makes a point of excluding PCIAT to avoid leakage — the analogous
point here is **subject-grouped splitting**).

---

## 9. Documented discrepancy (note in the report, low impact)

The dataset's bundled `instruction_DM2_timeseries.txt` says each series is the
"average signal in **30-minute** intervals" — which for 200 steps would span
~4.2 days. The Module-0 notebook instead assumes ~7.2-min epochs (a single
subject-day). **Verified fact:** `weekday`, `quarter`, and `relative_date_PCIAT`
are each **constant within every one of the 4437 series**, i.e. **one series = one
subject-day**. This supports the single-day reading and contradicts the literal
30-min figure in the instruction text. The exact epoch duration doesn't affect the
classification methodology, but state the one-series-per-subject-day fact clearly
(it underpins the grouped-split argument).

---

## 10. Open decisions for the planner

1. **Training vs evaluation granularity:** train **per-series** (4252, grouped CV)
   [default — more data]; **evaluate per-subject** (326, pool a child's days)
   [recommended headline, §3.3]. Decide the pooling rule (mean-proba vs majority).
2. **Prolific-subject dominance (§3.2):** apply `1/n_days` series weighting and/or
   cap days-per-subject — and reuse the cap as the DTW subsample (item 5). Decide
   the cap `k`.
3. **Normalization:** per-channel train-fit scaling [recommended] vs per-series
   z-norm — justify w.r.t. preserving orientation/intensity signal.
4. **Multivariate strategy per method:** native multivariate vs channel-concat vs
   channel-subset (e.g. enmo+anglez).
5. **DTW cost control:** band width tuning, PAA/SAX downsampling, and/or subsample
   size (day-capping, item 2) if full pairwise DTW is infeasible on 4252 series.
6. **Which "other method"(s):** ROCKET family (no DL needed, recommended) and/or
   MUSE; deep nets only if a backend installs on Python 3.14.
7. **Whether to include** the optional sequential-pattern-mining and/or an XAI
   paragraph.
8. **Library:** confirm `aeon` (or fallback) installs on Python 3.14 before
   committing the plan; otherwise pick the subset of methods whose libs install.
