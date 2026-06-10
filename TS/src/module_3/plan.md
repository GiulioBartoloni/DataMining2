# Module 3 — Time Series Classification (`sii_binary`): Implementation Plan

> Planning document only — **no model code yet**. Scope: binary TSC of `sii_binary`
> from `TS/processed_data/ts_preprocessed.pkl.gz` (4252 series × 200 × 5ch, 326
> subjects). Authoritative source: `research.md`. Motifs/discords & clustering are
> out of scope. Every decision below is resolved with a concrete choice; deviations
> from `research.md` are flagged inline.

All dependency claims below were **verified empirically** on this machine (isolated
Python 3.14.5 venv, `aeon==1.4.0`), not assumed — see §1.

---

## 0. Decisions resolved (research §10) — quick reference

| # | Open decision | **Chosen** | Why (short) |
|---|---|---|---|
| 1 | Train vs eval granularity | Train **per-series**; **evaluate per-subject** (pool a child's days by **mean predicted probability**), headline = subject-level | `sii` is a subject attribute; series-level prevalence is distorted (32.9 % vs 36.8 %). Mean-proba supports subject-level ROC/PR. |
| 2 | Prolific-subject dominance | **Day-cap k = 10** random days/subject (seed 42) → **2178 series**, applied uniformly to all methods. `1/n_days` weighting used only in a full-data robustness check. | Cap trims only the prolific tail (median = 11.5), curbs neighbour/fingerprint dominance for *all* methods, and doubles as the DTW subsample (consistent comparison set). |
| 3 | Normalization | **Per-channel z-score, statistics fit on the training fold only** (not per-series z-norm) | Preserves limb-orientation (X/Y/Z mean) and activity-intensity (`enmo` level) signal while equalizing the ~2-orders-of-magnitude channel-scale gap. |
| 4 | Multivariate strategy | **Native multivariate** for every method; **channel-subset ablation** (`enmo`+`anglez`) for KNN-DTW and MiniRocket | aeon handles MV natively; concat would inflate DTW length 5×. enmo+anglez are the classic actigraphy signals. |
| 5 | DTW cost control | Sakoe-Chiba **band tuned over {0.02, 0.05, 0.10}**; **precompute the full banded pairwise matrix once per window** on the capped set; PAA→100 kept as a fallback only | Precompute makes all CV folds + all `k` free. ~2.4M pairs × 0.16 ms ≈ 6 min/window; 3 windows ≈ 20 min total — PAA not needed at k = 10. |
| 6 | "Other method(s)" | **MiniRocket (primary)** + **MUSE (secondary)**. **No deep nets.** | TF has no cp314 wheels (aeon DL is TF-based); both chosen methods need no DL backend and exceed the "≥1" requirement. |
| 7 | SPM / XAI | **Skip SPM** (optional, time-risky). **Include a short shapelet-interpretation (XAI) paragraph** as the interpretability hook tying to Module 2. | Shapelet plots already deliver intrinsic interpretability; SPM adds scope for little marginal value. **← the one decision worth your override (see §9).** |
| 8 | Library | **`aeon 1.4.0` in a dedicated Module-3 venv** (`TS/.venv-ts`) | Verified to install & run on 3.14, but it downgrades pandas/numpy/sklearn — isolating protects the tabular modules' environment. |

---

## 1. Dependency step (DONE — verified, not assumed)

**Environment:** repo `.venv` is Python **3.14.5** with pandas 3.0.2 / numpy 2.4.6 /
sklearn 1.8.0 (the tabular stack). `aeon` is **not** installed.

**What I verified by actually installing `aeon` into a throwaway 3.14 venv and running
the estimators:**

1. `aeon==1.4.0` **installs and runs on Python 3.14.5.** ✅
2. **It forces downgrades:** `pandas 3.0→2.3.3`, `numpy 2.4→2.3.5`, `scikit-learn
   1.8→1.7.2`, `numba 0.65→0.63.1`, `llvmlite 0.47→0.46`. Root cause: `numba`
   (aeon's JIT accel for distances/shapelets) has **no numpy-2.4 support yet**.
   → Installing into the shared `.venv` would silently change the tabular modules'
   numeric stack. **Decision: dedicated venv** (§0.8).
3. **Smoke-tested on dummy `(N,5,200)` data — all required estimators work:**
   `KNeighborsTimeSeriesClassifier(distance="euclidean"|"manhattan"|"dtw")` ✅,
   `RocketClassifier`/`MiniRocketClassifier`/`MultiRocketClassifier` ✅,
   `RDSTClassifier` ✅, `RandomShapeletTransform` ✅ (exposes a `.shapelets` list for
   plotting), `MUSE` ✅, `sklearn.StratifiedGroupKFold` ✅.
4. **⚠️ `float32` lands aeon's numba shapelet kernels in a `TypingError`** ("Cannot
   unify array(float64) and array(float32)"). The dataset is `float32`. **The adapter
   MUST cast channels to `float64`.** (Verified: float64 fixes it.) — *not mentioned
   in research.md; a concrete, load-bearing gotcha.*
5. **Deep-learning backends on 3.14:** `tensorflow` has **no cp314 wheels** (stable
   tops out at cp313) → aeon's TF-based DL classifiers (InceptionTime, CNN, LSTM-FCN)
   are **unavailable**. `torch==2.12.0` *does* resolve on 3.14, but aeon's DL module
   doesn't use it. → **Fallback per research §7 confirmed: satisfy "≥1 other method"
   with the ROCKET family + MUSE** (no DL needed). A hand-rolled PyTorch 1D-CNN is a
   possible stretch goal but is out of scope.

**Action for implementation (first cell / one-time setup):**
```bash
uv venv --python 3.14 TS/.venv-ts
uv pip install --python TS/.venv-ts/bin/python \
    aeon matplotlib seaborn scikit-learn          # aeon pins compatible numpy/pandas/sklearn
TS/.venv-ts/bin/python -m ipykernel install --user --name dm2-ts --display-name "DM2-TS (aeon)"
```
Register the kernel so the notebook runs against the isolated stack. Pin versions in a
short `TS/src/module_3/requirements-ts.txt` for reproducibility. *(A working install
already exists at `/tmp/aeon_probe` from verification — disposable; the real venv is
`TS/.venv-ts`.)*

**Risk note:** if a future `aeon` patch breaks on 3.14, fallback chain is
`tslearn` (KNN-DTW + shapelets) + `pyts` (ROCKET/BOSS) — both lighter, also numba/
numpy-sensitive. Verify before switching.

---

## 2. Data adapter — `list[DataFrame] → (X, y, groups)`

Single function, lives in the first code section; **the contract every model consumes.**

```
load ts_preprocessed.pkl.gz  ->  list of 4252 DataFrames
for each df:
    X_i = df[['X','Y','Z','enmo','anglez']].to_numpy().T   # (5, 200)
    y_i = df['sii_binary'].iloc[0]                          # scalar 0/1
    g_i = df['id'].iloc[0]                                  # subject id
X      = np.stack(X_i).astype(np.float64)   # (4252, 5, 200)  ← float64, NOT float32 (§1.4)
y      = np.array(y_i)                       # (4252,)
groups = np.array(g_i)                       # (4252,)
```
Assertions to bake in: `X.shape == (4252,5,200)`, no NaNs, `len(set(g))==326`, label
100 % consistent within each group (re-verify — it holds in the raw data).

**Day-cap (decision §0.2):** `rng = np.random.default_rng(42)`; for each subject keep
`min(n_days, 10)` random series → indices for the **capped modelling set** (≈2178,
5ch, 200). Keep the full set around only for the §6 robustness check. Store
`n_days[id]` for the `1/n_days` weighting option.

---

## 3. Split & nested grouped CV (research §3.1 — non-negotiable)

- **Outer:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on
  `(y, groups=id)` over the **capped** set. Report **mean ± std across the 5 outer
  folds** (effective n ≈ 326 subjects → single split is noisy, §3.2).
- **Inner (tuning):** `StratifiedGroupKFold(n_splits=3)` inside
  `RandomizedSearchCV(..., cv=inner, scoring='f1_macro')`, **groups passed through**
  so a child's days never straddle inner folds.
- **Scaler fit on train fold only**, per channel: compute mean/std over
  `(cases_train, timepoints)` per channel, apply to train+test of that fold. (Custom
  transformer or manual; aeon has `Normalizer`/`channel` scalers but per-channel
  train-fit is cleanest done by hand to guarantee no leakage.)
- **Subject-level evaluation (headline, §3.3):** within each outer test fold, predict
  per-series probabilities → group by `id` → **mean probability** per subject →
  threshold (tuned on inner CV) → metrics over that fold's test subjects. Pool
  fold-level subject metrics into mean ± std. Series-level metrics reported alongside.
- **⚠️ Implementation risk:** sklearn `RandomizedSearchCV` over **3-D `X`** + aeon
  estimators can have friction (array checks, `groups` plumbing). Mitigation: a thin
  **manual grouped-CV loop** wrapper (clone → fit on `X[tr]` → score) as fallback;
  for KNN-DTW use the **precomputed distance matrix** path (§5) which sidesteps it
  entirely.

**Baselines (research §6):** `DummyClassifier(strategy='most_frequent')` (subject-acc
≈ 0.632), and a **naive ungrouped** `StratifiedKFold` run of one model (e.g. KNN-DTW)
to **quantify the leakage gap** — report both as a methodological exhibit.

---

## 4. Notebook layout — `TS/src/module_3/classification.ipynb`

Sections in order (each model = "Model & tuning" + "Results" paragraphs, mirroring the
report's Module 2 prose structure):

| # | Section | Content |
|---|---|---|
| 0 | **Setup** | venv/kernel note, imports, `random_state=42`, helper imports |
| 1 | **Load & adapter** | §2: list→`(X,y,groups)` float64, assertions, day-cap |
| 2 | **EDA for the report** | class balance (series vs subject), series/subject histogram (leakage motivation), example series per class per channel, channel-scale table |
| 3 | **CV / eval harness** | §3: outer/inner `StratifiedGroupKFold`, per-channel train-fit scaler, subject-pooling fn, metrics fn, baseline + naive-split exhibit |
| 4 | **KNN (Euclidean / Manhattan / DTW)** | §5 |
| 5 | **Shapelets** | §5.3 — retrieve + **plot/analyse** + classify |
| 6 | **MiniRocket** (primary other) | §5.4 |
| 7 | **MUSE** (secondary other) | §5.4 |
| 8 | **Comparison & discussion** | §7: comparison table, grouped bar chart, ROC/PR overlays, confusion matrices, leakage-gap exhibit |
| 9 | **(optional) Shapelet-XAI paragraph** | interpretation of the discriminative shapes (§0.7) |

---

## 5. Models — estimators, grids, multivariate strategy

### 5.1 KNN (the required distance comparison)
- **Estimator:** `KNeighborsTimeSeriesClassifier` (native multivariate).
- **Distances:** `euclidean`, `manhattan`, `dtw` (DTW with Sakoe-Chiba band via
  `distance_params={"window": w}`). DTW uses **dependent** multivariate distance
  (single warping path across channels) — aeon default for MV `dtw`.
- **Grid:** `n_neighbors ∈ {1,3,5,7,9}`; for DTW also `window ∈ {0.02,0.05,0.10}`
  (research §5.1: optimum ~2–8 % of length). `weights ∈ {uniform, distance}`.
- **DTW cost path (decision §0.5):** for each `window`, **precompute the full
  symmetric pairwise DTW matrix once** on the capped set with
  `aeon.distances.dtw_pairwise_distance(X, window=w)` (~2.4M pairs ≈ 6 min); then KNN
  for any fold/`k` is index-slicing the matrix + a vote — effectively free. (Euclidean/
  Manhattan are cheap; compute directly.)
- **Ablation:** rerun best DTW config on the **`enmo`+`anglez` channel subset** to test
  whether the actigraphy pair alone carries the signal.
- **Report (research §6):** effect of `k`, **Euclidean/Manhattan vs DTW**, band width,
  and the **subject-neighbour bias** discussion (prolific subjects dominate the vote —
  why the cap matters here most).

### 5.2 Shapelets (retrieve + analyse — the interpretability deliverable)
- **Retrieve & plot (the "analyse the shapelets" requirement):**
  `RandomShapeletTransform(n_shapelet_samples=..., max_shapelets=K, random_state=42)`
  on a training fold. Its `.shapelets` list gives, per shapelet, the tuple
  `(info_gain, length, position, channel, dilation, series_index, data_array)` —
  **plot the top-N by information gain**, annotated with their channel and which class
  they discriminate, and **overlay each on a matching example series** to show *where*
  it fires (subsequence-distance localization).
- **Classify:** `RandomShapeletTransform` → `RidgeClassifierCV` (or `RandomForest`) on
  the `N×K` distance table = the textbook **Shapelet Transform pipeline** (research
  §5.3). Native multivariate (shapelets carry a channel index).
- **Also in the comparison table:** `RDSTClassifier` (Random Dilated Shapelet
  Transform) as the **fast/strong modern shapelet classifier** — gives a competitive
  accuracy number to sit next to the interpretable transform.
  *(Avoid `ShapeletTransformClassifier`'s RotationForest backend — too slow.)*
- **Grid:** `max_shapelets ∈ {50,100}`, `n_shapelet_samples` modest for time;
  RDST `max_shapelets ∈ {1000(default), 2000}`.
- **Guard (research §3.2.3):** check the top shapelets' `series_index` aren't all from
  one prolific subject (a **subject fingerprint masquerading as a class shapelet**) —
  the cap mitigates this; verify and note.

### 5.3 MiniRocket — primary "other method"
- **Estimator:** `MiniRocketClassifier(n_kernels=10000, random_state=42)` (native MV;
  fixed kernels + PPV → ridge). Near-top accuracy at a fraction of the cost (research
  §5.4 benchmark takeaway).
- **Grid:** essentially the ridge `alpha` (built-in `RidgeClassifierCV`); optionally
  `n_kernels ∈ {10000, 20000}`. `class_weight='balanced'` on the ridge head.
- **For ROC/PR/subject-pooling** we need probabilities: use `RocketClassifier` with a
  logistic head, or wrap MiniRocket transform + `LogisticRegression(class_weight=
  'balanced')` so `predict_proba` exists. **Decision: MiniRocket transform +
  LogisticRegression** (gives calibrated-ish probabilities for the mean-pooling rule).
- **Ablation:** channel-subset (`enmo`+`anglez`) vs all-5.
- **Report:** accuracy vs training-time trade-off; fingerprinting/overfitting controls
  (high capacity → why grouped eval + cap matter, research §3.2.4).

### 5.4 MUSE — secondary "other method"
- **Estimator:** `MUSE(random_state=42)` (WEASEL+MUSE, native multivariate dictionary
  model, explicitly listed in the guidelines). Has `predict_proba`.
- **Grid:** keep light — `MUSE` is slow (~18 s on tiny data incl. JIT); tune at most
  `window_inc`/`use_first_order_differences` if time allows, else defaults.
- Provides method diversity (dictionary view) for the comparison story:
  **baseline (KNN) → interpretable (shapelets) → strong/fast kernel (MiniRocket) →
  dictionary (MUSE).**

### Imbalance handling (all methods, research §6)
`class_weight='balanced'` wherever supported (mild 63/37 subject split). **Threshold
tuning:** after mean-proba pooling, pick the decision threshold maximizing F1 on inner
CV, apply at subject level. `1/n_days` series weighting only in the §6 robustness check.

---

## 6. Evaluation & comparison artifacts

**Metrics (binary, positive = problematic = 1):** accuracy, precision, recall, **F1**,
**balanced accuracy / macro-F1**, **ROC-AUC**, **PR-AUC** — reported **per fold as
mean ± std**, at **subject level (headline)** and series level. Per-class breakdown.

**Tables (match report conventions — `\label{tab:...}`, `\small`):**
- `tab:ts_clf` — **main comparison** (one row per method): `CV F1 | Subj Acc | Subj
  F1 | Subj AUC | Series F1` (mirrors `tab:cmi_clf`). Bold the winner per column.
- `tab:ts_params` — best hyperparameters per method (mirrors `tab:cmi_clf_params`).
- `tab:ts_baseline` — Dummy + **naive-ungrouped vs grouped** leakage gap (the exhibit).

**Figures (`\imgph{...}` placeholders, like the tabular report):**
- `fig:ts_data` — class balance (series vs subject) + series-per-subject histogram
  (leakage motivation) + example series per class.
- `fig:ts_knn` — KNN sweep: F1 vs `k` for Euclidean/Manhattan/DTW; DTW F1 vs band width.
- `fig:ts_shapelets` — **top discriminative shapelets plotted** + each overlaid on a
  matching example series (the analysis deliverable).
- `fig:ts_roc` — subject-level ROC + PR curves, all methods overlaid.
- `fig:ts_cm` — confusion matrices grid (subject level) per method.
- `fig:ts_compare` — grouped bar chart (Subj F1 / AUC / balanced-acc) across methods.

**Report section to draft (new — there is none yet):** a `\section{Time Series
Classification}` with subsections per §4, opening with a **"Common protocol"**
paragraph (capped set, subject-grouped nested CV, per-channel train-fit scaling,
subject-level headline, `random_state=42`) — directly analogous to the Module 2
"Common protocol" paragraph — and a **leakage discussion** paragraph that is the
time-series analogue of the tabular report's PCIAT-exclusion argument (here:
subject-grouped splitting + why naive splits leak).

---

## 7. Ordered task list (effort / risk)

| # | Task | Effort | Risk | Notes |
|---|---|---|---|---|
| 1 | Create `TS/.venv-ts`, install aeon, register kernel, pin reqs | 0.5 h | **Low** (verified) | numpy/pandas downgrade is contained to this venv |
| 2 | Data adapter + assertions + **float64** + day-cap(k=10) | 1 h | Low | float64 is mandatory (§1.4) |
| 3 | EDA cells + report data figures | 1.5 h | Low | reuse for `fig:ts_data` |
| 4 | CV/eval harness: nested grouped CV, per-channel scaler, subject pooling, metrics, baselines + naive-split exhibit | 3 h | **Med** | sklearn×aeon 3-D friction (§3); manual-loop fallback ready |
| 5 | KNN Euclidean/Manhattan + **precomputed DTW matrices** (3 windows) + sweep + channel-subset | 3 h | **Med** | DTW ≈ 20 min compute; precompute is the cost guard |
| 6 | Shapelets: RandomShapeletTransform retrieve + **plots/analysis** + Ridge head + RDST | 3 h | **Med** | RDST JIT warmup slow; fingerprint guard (§5.2) |
| 7 | MiniRocket (+ LogisticRegression head) + channel-subset ablation | 1.5 h | Low | fast |
| 8 | MUSE | 1.5 h | **Med** | slow; keep grid tiny |
| 9 | Comparison tables + figures + ROC/PR/CM | 2 h | Low | |
| 10 | Draft `\section{Time Series Classification}` for `report_dm2.tex` | 2.5 h | Low | match Module 2 prose + `\imgph` conventions |
| 11 | (optional) Shapelet-XAI paragraph | 0.5 h | Low | §0.7 |
| 12 | (optional, only if asked) SPM via SAX | +3 h | Med | **not in baseline plan** (§9) |

**Total baseline ≈ 21 h.** Two dominant risks, both mitigated: **(a) Python-3.14
library risk** — retired by actually installing & running aeon (§1); residual = a
future aeon patch breaking, fallback tslearn/pyts. **(b) DTW cost** — retired by
day-capping + precomputed banded matrices (~20 min, not hours).

---

## 8. Deviations from / additions to `research.md` (flagged)

- **float64 adapter** — research.md says the channels are float32 and implies stacking
  as-is; aeon's numba shapelet kernels **crash on float32**. The adapter must cast to
  float64. *(New, verified constraint.)*
- **Dedicated venv** — research.md recommends "add aeon" to the existing venv; doing so
  **downgrades pandas 3.0→2.3 / numpy 2.4→2.3 / sklearn 1.8→1.7** and would change the
  tabular modules' stack. I isolate Module 3 instead. *(Refinement of §7.)*
- **Day-cap chosen as the primary mitigation** (k=10), with `1/n_days` weighting demoted
  to a robustness check, rather than research §3.2's "and/or". Reason: the cap is a
  *single* lever that simultaneously curbs prolific-subject dominance for all methods,
  keeps a consistent comparison set, and bounds DTW cost — combining both mitigations is
  redundant. *(A choice within §10.2's option space, not a contradiction.)*
- **Precomputed DTW distance matrix** — research.md lists band/PAA/subsample; I add the
  precompute-once trick, which is what actually makes the full CV×k sweep cheap.
- **No deep net** — research.md leaves DL conditional on backend install; I confirm
  it's **blocked** (TF no cp314 wheel) and close it. ROCKET+MUSE cover the requirement.

---

## 9. The one decision worth your input (non-blocking)

**Sequential Pattern Mining (research §10.7) is optional.** My plan **skips it** and
instead spends that interpretability budget on the **shapelet-analysis paragraph**
(which is already a hard requirement and a natural XAI tie-in to Module 2). SPM would
add ~3 h (SAX discretization + a frequent-sequence miner not currently installed) for
marginal narrative gain.

If you'd rather **include SPM** (e.g. to broaden method coverage or because the
guidelines reward the optional task), say so and I'll add: per-channel SAX → frequent
sequential-pattern mining → patterns correlated with the label, as section 8.5. Absent
a reply, I proceed **without SPM** and **with** the shapelet-XAI paragraph.

Every other §10 decision is resolved above with a justified default and needs no input.
