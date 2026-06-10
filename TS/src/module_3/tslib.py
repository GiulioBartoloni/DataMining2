"""Module 3 — Time Series Classification: shared foundation.

Data adapter, day-capping, per-channel scaling, subject-level pooling, metrics,
and a manual nested grouped-CV harness. Kept dependency-light (numpy / sklearn)
so it can be reused verbatim as notebook cells.

Run directly (`python tslib.py`) for a self-check on the real dataset.
"""
from __future__ import annotations

import gzip
import os
import pickle
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
CHANNELS = ["X", "Y", "Z", "enmo", "anglez"]
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "..", "processed_data", "ts_preprocessed.pkl.gz")
CACHE_DIR = os.path.join(HERE, "cache")
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Data adapter
# --------------------------------------------------------------------------- #
def load_raw(path: str = DATA_PATH):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def build_arrays(data):
    """list[DataFrame] -> (X float64 (N,5,200), y (N,), groups (N,)).

    float64 is mandatory: aeon's numba shapelet kernels raise a TypingError on
    float32 (verified).
    """
    X = np.stack([df[CHANNELS].to_numpy().T for df in data]).astype(np.float64)
    y = np.array([int(df["sii_binary"].iloc[0]) for df in data])
    groups = np.array([int(df["id"].iloc[0]) for df in data])
    # contract checks
    assert X.shape == (len(data), 5, 200), X.shape
    assert not np.isnan(X).any()
    for sid in np.unique(groups):
        assert len(np.unique(y[groups == sid])) == 1, f"label not consistent for {sid}"
    return X, y, groups


def day_cap(X, y, groups, k: int = 10, seed: int = SEED):
    """Keep <=k random series per subject. Returns capped (X,y,groups,n_days).

    n_days maps each kept row to its subject's ORIGINAL series count (for the
    optional 1/n_days weighting robustness check).
    """
    rng = np.random.default_rng(seed)
    keep = []
    counts = {sid: int((groups == sid).sum()) for sid in np.unique(groups)}
    for sid in np.unique(groups):
        idx = np.where(groups == sid)[0]
        if len(idx) > k:
            idx = rng.choice(idx, size=k, replace=False)
        keep.extend(idx.tolist())
    keep = np.sort(np.array(keep))
    n_days = np.array([counts[g] for g in groups[keep]])
    return X[keep], y[keep], groups[keep], n_days


# --------------------------------------------------------------------------- #
# 2. Per-channel scaling (global, label-independent)
# --------------------------------------------------------------------------- #
def channel_scaler(X):
    """Return (mean, std) per channel over (cases, timepoints). Shape (1,C,1)."""
    mean = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2), keepdims=True)
    std[std == 0] = 1.0
    return mean, std


def apply_scaler(X, stats):
    mean, std = stats
    return (X - mean) / std


# --------------------------------------------------------------------------- #
# 3. Subject-level pooling + metrics
# --------------------------------------------------------------------------- #
def pool_subject(p, y, groups):
    """Pool per-series positive-class proba -> per-subject by mean.

    Returns (subj_ids, subj_y, subj_p).
    """
    sids = np.unique(groups)
    sy = np.array([y[groups == s][0] for s in sids])
    sp = np.array([p[groups == s].mean() for s in sids])
    return sids, sy, sp


def binary_metrics(y_true, p, thr: float = 0.5):
    yhat = (p >= thr).astype(int)
    out = {
        "acc": accuracy_score(y_true, yhat),
        "bal_acc": balanced_accuracy_score(y_true, yhat),
        "f1": f1_score(y_true, yhat, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, yhat, average="macro", zero_division=0),
        "precision": precision_score(y_true, yhat, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, yhat, pos_label=1, zero_division=0),
    }
    # AUCs need both classes present
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, p)
        out["pr_auc"] = average_precision_score(y_true, p)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


# --------------------------------------------------------------------------- #
# 4. Nested grouped CV harness
# --------------------------------------------------------------------------- #
@dataclass
class CVResult:
    name: str
    fold_subject: list = field(default_factory=list)   # per-fold metric dicts (subject level)
    fold_series: list = field(default_factory=list)    # per-fold metric dicts (series level)
    best_params: list = field(default_factory=list)    # per-fold chosen params
    oof_subject: dict = field(default_factory=dict)    # {'y':, 'p':, 'sid':}
    oof_series: dict = field(default_factory=dict)      # {'y':, 'p':}
    cv_score: float = np.nan                            # mean inner score of selected models

    def summary(self):
        def agg(folds, key):
            v = np.array([f[key] for f in folds], dtype=float)
            return np.nanmean(v), np.nanstd(v)
        s = {}
        for key in ["acc", "bal_acc", "f1", "f1_macro", "roc_auc", "pr_auc",
                    "precision", "recall"]:
            m, sd = agg(self.fold_subject, key)
            s[f"subj_{key}"] = m
            s[f"subj_{key}_std"] = sd
        for key in ["f1", "roc_auc", "bal_acc"]:
            m, sd = agg(self.fold_series, key)
            s[f"ser_{key}"] = m
        s["cv_score"] = self.cv_score
        return s


def _inner_score(p, ytrue, scoring):
    if scoring == "roc_auc":
        if len(np.unique(ytrue)) < 2:
            return np.nan
        return roc_auc_score(ytrue, p)
    yhat = (p >= 0.5).astype(int)
    return f1_score(ytrue, yhat, average="macro", zero_division=0)


def nested_cv(
    fit_predict,
    param_grid,
    X,
    y,
    groups,
    name,
    scale=True,
    n_outer=5,
    n_inner=3,
    scoring="f1_macro",
    seed=SEED,
    verbose=True,
):
    """Manual nested grouped CV.

    fit_predict(params, Xtr, ytr, Xte) -> p_pos (proba of class 1 for Xte rows).
    param_grid: list of param dicts (already expanded). Inner grouped CV picks the
    combo with the best mean inner score; that combo is refit on the full outer
    train and scored on the outer test (series + subject level).
    """
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    res = CVResult(name=name)
    oof_y, oof_p, oof_sid = [], [], []
    soof_y, soof_p = [], []
    inner_scores_sel = []

    for fold, (tr, te) in enumerate(outer.split(X, y, groups)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        gtr, gte = groups[tr], groups[te]
        if scale:
            stats = channel_scaler(Xtr)
            Xtr_s, Xte_s = apply_scaler(Xtr, stats), apply_scaler(Xte, stats)
        else:
            Xtr_s, Xte_s = Xtr, Xte

        # ---- inner grouped CV model selection ----
        if len(param_grid) == 1:
            best = param_grid[0]
            best_inner = np.nan
        else:
            inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
            best, best_inner = None, -np.inf
            for params in param_grid:
                sc = []
                for itr, ite in inner.split(Xtr_s, ytr, gtr):
                    p = fit_predict(params, Xtr_s[itr], ytr[itr], Xtr_s[ite])
                    sc.append(_inner_score(p, ytr[ite], scoring))
                msc = np.nanmean(sc)
                if msc > best_inner:
                    best_inner, best = msc, params
        inner_scores_sel.append(best_inner)

        # ---- refit on outer train, predict outer test ----
        p = fit_predict(best, Xtr_s, ytr, Xte_s)
        res.best_params.append(best)

        # series level
        res.fold_series.append(binary_metrics(yte, p))
        soof_y.append(yte); soof_p.append(p)
        # subject level
        sids, sy, sp = pool_subject(p, yte, gte)
        res.fold_subject.append(binary_metrics(sy, sp))
        oof_y.append(sy); oof_p.append(sp); oof_sid.append(sids)

        if verbose:
            fs = res.fold_subject[-1]
            print(f"  [{name}] fold {fold}: subj F1={fs['f1']:.3f} "
                  f"AUC={fs['roc_auc']:.3f} balacc={fs['bal_acc']:.3f} "
                  f"(best={best})")

    res.cv_score = float(np.nanmean(inner_scores_sel))
    res.oof_subject = {"y": np.concatenate(oof_y), "p": np.concatenate(oof_p),
                       "sid": np.concatenate(oof_sid)}
    res.oof_series = {"y": np.concatenate(soof_y), "p": np.concatenate(soof_p)}
    return res


# --------------------------------------------------------------------------- #
# self-check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    data = load_raw()
    X, y, groups = build_arrays(data)
    print(f"full: X{X.shape} dtype={X.dtype} subjects={len(np.unique(groups))} "
          f"prev={y.mean():.3f}")
    Xc, yc, gc, nd = day_cap(X, y, groups, k=10)
    print(f"capped k=10: X{Xc.shape} subjects={len(np.unique(gc))} prev={yc.mean():.3f}")

    # quick sanity: 1-NN euclidean via aeon on capped set, one CV pass
    from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier

    def fp(params, Xtr, ytr, Xte):
        clf = KNeighborsTimeSeriesClassifier(distance="euclidean", **params)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)
        return proba[:, list(clf.classes_).index(1)]

    res = nested_cv(fp, [{"n_neighbors": 5}], Xc, yc, gc, name="KNN-Euclid(self-check)")
    print("summary:", {k: round(v, 3) for k, v in res.summary().items()
                       if not k.endswith("_std")})
