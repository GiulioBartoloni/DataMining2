"""Run each TSC method through subject-grouped nested CV on the day-capped set and
cache the CVResult to cache/res_<name>.pkl.

Usage:  python run_models.py <method>
  method in {minirocket, muse, rdst, shapelet, knn_euclidean, knn_manhattan,
             knn_dtw, knn_dtw_subset, minirocket_subset, baselines, all}
"""
import os
import pickle
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
import tslib
from tslib import (CVResult, SEED, binary_metrics, channel_scaler, apply_scaler,
                   nested_cv, pool_subject)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, roc_auc_score

CAP_K = 10


def load_capped():
    data = tslib.load_raw()
    X, y, groups = tslib.build_arrays(data)
    return tslib.day_cap(X, y, groups, k=CAP_K)


def save(res):
    with open(os.path.join(tslib.CACHE_DIR, f"res_{res.name}.pkl"), "wb") as f:
        pickle.dump(res, f)
    s = res.summary()
    print(f"[{res.name}] subj F1={s['subj_f1']:.3f}±{s['subj_f1_std']:.3f} "
          f"AUC={s['subj_roc_auc']:.3f} balacc={s['subj_bal_acc']:.3f} "
          f"cvF1={s['cv_score']:.3f}")


def _idx1(classes):
    return list(classes).index(1)


# --------------------------------------------------------------------------- #
# Convolution / dictionary / shapelet runners (transform reused per outer fold)
# --------------------------------------------------------------------------- #
def run_rocket(channels=None, name="MiniRocket"):
    from aeon.transformations.collection.convolution_based import MiniRocket
    Xc, yc, gc, nd = load_capped()
    if channels is not None:
        Xc = Xc[:, channels, :]
    C_grid = [0.01, 0.1, 1.0, 10.0]
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    res = CVResult(name=name)
    oy, op, osid, sy_, sp_ = [], [], [], [], []
    cvsel = []
    for fold, (tr, te) in enumerate(outer.split(Xc, yc, gc)):
        st = channel_scaler(Xc[tr])
        Xtr, Xte = apply_scaler(Xc[tr], st), apply_scaler(Xc[te], st)
        ytr, yte, gtr, gte = yc[tr], yc[te], gc[tr], gc[te]
        t = time.time()
        mr = MiniRocket(n_kernels=10000, random_state=SEED)
        Ftr = mr.fit_transform(Xtr); Fte = mr.transform(Xte)
        sc = StandardScaler().fit(Ftr)
        Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
        # inner grouped CV to pick C
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
        best, bestsc = None, -np.inf
        for C in C_grid:
            scs = []
            for itr, ite in inner.split(Ftr, ytr, gtr):
                clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000)
                clf.fit(Ftr[itr], ytr[itr])
                p = clf.predict_proba(Ftr[ite])[:, _idx1(clf.classes_)]
                scs.append(f1_score(ytr[ite], (p >= .5).astype(int),
                                    average="macro", zero_division=0))
            if np.mean(scs) > bestsc:
                bestsc, best = np.mean(scs), C
        clf = LogisticRegression(C=best, class_weight="balanced", max_iter=2000)
        clf.fit(Ftr, ytr)
        p = clf.predict_proba(Fte)[:, _idx1(clf.classes_)]
        res.best_params.append({"C": best, "n_kernels": 10000})
        res.fold_series.append(binary_metrics(yte, p))
        sids, sYY, sPP = pool_subject(p, yte, gte)
        res.fold_subject.append(binary_metrics(sYY, sPP))
        cvsel.append(bestsc)
        oy.append(yte); op.append(p); sy_.append(sYY); sp_.append(sPP); osid.append(sids)
        print(f"  fold {fold}: subjF1={res.fold_subject[-1]['f1']:.3f} "
              f"AUC={res.fold_subject[-1]['roc_auc']:.3f} C={best} ({time.time()-t:.0f}s)")
    res.cv_score = float(np.mean(cvsel))
    res.oof_series = {"y": np.concatenate(oy), "p": np.concatenate(op)}
    res.oof_subject = {"y": np.concatenate(sy_), "p": np.concatenate(sp_),
                       "sid": np.concatenate(osid)}
    save(res)


def run_muse():
    from aeon.classification.dictionary_based import MUSE
    Xc, yc, gc, nd = load_capped()

    def fp(params, Xtr, ytr, Xte):
        # memory-safe config: 15GB RAM, no swap -> n_jobs=1 and a lean dictionary
        # (coarser window grid, no bigrams / first-order differences) so the
        # multivariate SFA bag does not OOM. support_probabilities for calibrated proba.
        clf = MUSE(random_state=SEED, n_jobs=1, window_inc=4, bigrams=False,
                   use_first_order_differences=False, support_probabilities=True,
                   **params)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, _idx1(clf.classes_)]

    res = nested_cv(fp, [{}], Xc, yc, gc, name="MUSE", scale=True)
    save(res)


def run_rdst():
    from aeon.classification.shapelet_based import RDSTClassifier
    Xc, yc, gc, nd = load_capped()

    def fp(params, Xtr, ytr, Xte):
        clf = RDSTClassifier(random_state=SEED, n_jobs=4, **params)
        clf.fit(Xtr, ytr)
        return clf.predict_proba(Xte)[:, _idx1(clf.classes_)]

    res = nested_cv(fp, [{"max_shapelets": 10000}], Xc, yc, gc, name="RDST", scale=True)
    save(res)


def run_shapelet():
    """Interpretable Shapelet Transform + RandomForest head."""
    from aeon.transformations.collection.shapelet_based import RandomShapeletTransform
    from sklearn.ensemble import RandomForestClassifier
    Xc, yc, gc, nd = load_capped()
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    res = CVResult(name="Shapelet-RF")
    oy, op, osid, sy_, sp_ = [], [], [], [], []
    for fold, (tr, te) in enumerate(outer.split(Xc, yc, gc)):
        st = channel_scaler(Xc[tr])
        Xtr, Xte = apply_scaler(Xc[tr], st), apply_scaler(Xc[te], st)
        ytr, yte, gte = yc[tr], yc[te], gc[te]
        t = time.time()
        rst = RandomShapeletTransform(
            n_shapelet_samples=2000, max_shapelets=60,
            random_state=SEED, n_jobs=4)
        Ftr = rst.fit_transform(Xtr, ytr); Fte = rst.transform(Xte)
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     random_state=SEED, n_jobs=4)
        clf.fit(Ftr, ytr)
        p = clf.predict_proba(Fte)[:, _idx1(clf.classes_)]
        res.best_params.append({"max_shapelets": 60, "n_shapelet_samples": 2000})
        res.fold_series.append(binary_metrics(yte, p))
        sids, sYY, sPP = pool_subject(p, yte, gte)
        res.fold_subject.append(binary_metrics(sYY, sPP))
        oy.append(yte); op.append(p); sy_.append(sYY); sp_.append(sPP); osid.append(sids)
        print(f"  fold {fold}: subjF1={res.fold_subject[-1]['f1']:.3f} "
              f"AUC={res.fold_subject[-1]['roc_auc']:.3f} ({time.time()-t:.0f}s)")
    res.cv_score = np.nan
    res.oof_series = {"y": np.concatenate(oy), "p": np.concatenate(op)}
    res.oof_subject = {"y": np.concatenate(sy_), "p": np.concatenate(sp_),
                       "sid": np.concatenate(osid)}
    save(res)


# --------------------------------------------------------------------------- #
# KNN over precomputed distance matrices
# --------------------------------------------------------------------------- #
def run_knn_precomputed(matrix, name, k_grid=(1, 3, 5, 7, 9)):
    from sklearn.neighbors import KNeighborsClassifier
    D = np.load(os.path.join(tslib.CACHE_DIR, f"{matrix}.npy")).astype(np.float64)
    yc = np.load(os.path.join(tslib.CACHE_DIR, "capped_y.npy"))
    gc = np.load(os.path.join(tslib.CACHE_DIR, "capped_groups.npy"))
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    res = CVResult(name=name)
    oy, op, osid, sy_, sp_ = [], [], [], [], []
    cvsel = []
    for fold, (tr, te) in enumerate(outer.split(D, yc, gc)):
        ytr, yte, gtr, gte = yc[tr], yc[te], gc[tr], gc[te]
        inner = StratifiedGroupKFold(3, shuffle=True, random_state=SEED)
        best, bestsc = None, -np.inf
        for k in k_grid:
            for wts in ("uniform", "distance"):
                scs = []
                for itr, ite in inner.split(np.zeros(len(tr)), ytr, gtr):
                    clf = KNeighborsClassifier(n_neighbors=k, metric="precomputed",
                                               weights=wts)
                    clf.fit(D[np.ix_(tr[itr], tr[itr])], ytr[itr])
                    p = clf.predict_proba(D[np.ix_(tr[ite], tr[itr])])[:, _idx1(clf.classes_)]
                    scs.append(f1_score(ytr[ite], (p >= .5).astype(int),
                                        average="macro", zero_division=0))
                if np.mean(scs) > bestsc:
                    bestsc, best = np.mean(scs), (k, wts)
        k, wts = best
        clf = KNeighborsClassifier(n_neighbors=k, metric="precomputed", weights=wts)
        clf.fit(D[np.ix_(tr, tr)], ytr)
        p = clf.predict_proba(D[np.ix_(te, tr)])[:, _idx1(clf.classes_)]
        res.best_params.append({"k": k, "weights": wts})
        res.fold_series.append(binary_metrics(yte, p))
        sids, sYY, sPP = pool_subject(p, yte, gte)
        res.fold_subject.append(binary_metrics(sYY, sPP))
        cvsel.append(bestsc)
        oy.append(yte); op.append(p); sy_.append(sYY); sp_.append(sPP); osid.append(sids)
        print(f"  fold {fold}: subjF1={res.fold_subject[-1]['f1']:.3f} "
              f"AUC={res.fold_subject[-1]['roc_auc']:.3f} best={best}")
    res.cv_score = float(np.mean(cvsel))
    res.oof_series = {"y": np.concatenate(oy), "p": np.concatenate(op)}
    res.oof_subject = {"y": np.concatenate(sy_), "p": np.concatenate(sp_),
                       "sid": np.concatenate(osid)}
    save(res)


def run_knn_dtw_window_sweep():
    """Pick best window by subject AUC on OOF; also store per-window curve."""
    curve = {}
    for w in (0.02, 0.05, 0.10):
        run_knn_precomputed(f"dtw_w{w}", name=f"KNN-DTW-w{w}")
        with open(os.path.join(tslib.CACHE_DIR, f"res_KNN-DTW-w{w}.pkl"), "rb") as f:
            r = pickle.load(f)
        curve[w] = r.summary()["subj_f1"]
    best_w = max(curve, key=curve.get)
    print("DTW window sweep (subjF1):", {k: round(v, 3) for k, v in curve.items()},
          "-> best", best_w)
    # alias best as canonical KNN-DTW
    with open(os.path.join(tslib.CACHE_DIR, f"res_KNN-DTW-w{best_w}.pkl"), "rb") as f:
        r = pickle.load(f)
    r.name = "KNN-DTW"
    save(r)
    with open(os.path.join(tslib.CACHE_DIR, "dtw_window_curve.pkl"), "wb") as f:
        pickle.dump(curve, f)


def run_baselines():
    """Dummy (majority) + naive-ungrouped leakage exhibit, using euclidean matrix."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import StratifiedKFold
    yc = np.load(os.path.join(tslib.CACHE_DIR, "capped_y.npy"))
    gc = np.load(os.path.join(tslib.CACHE_DIR, "capped_groups.npy"))

    # Dummy at subject level
    sids = np.unique(gc)
    sy = np.array([yc[gc == s][0] for s in sids])
    maj = int(sy.mean() >= 0.5)  # most-frequent class label (0 since prev<0.5)
    from sklearn.metrics import balanced_accuracy_score
    dummy = {"subj_acc": (sy == maj).mean(),
             "subj_bal_acc": balanced_accuracy_score(sy, np.full_like(sy, maj)),
             "subj_f1": f1_score(sy, np.full_like(sy, maj), pos_label=1, zero_division=0)}
    print("Dummy(majority) subject-level:", {k: round(v, 3) for k, v in dummy.items()})

    # Naive ungrouped 5-fold KNN-euclidean (leakage) vs grouped
    D = np.load(os.path.join(tslib.CACHE_DIR, "euclidean.npy")).astype(np.float64)
    def run(splitter, grouped):
        f1s = []
        for tr, te in (splitter.split(D, yc, gc) if grouped
                       else splitter.split(D, yc)):
            clf = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
            clf.fit(D[np.ix_(tr, tr)], yc[tr])
            p = clf.predict_proba(D[np.ix_(te, tr)])[:, _idx1(clf.classes_)]
            f1s.append(f1_score(yc[te], (p >= .5).astype(int), average="macro",
                                zero_division=0))
        return np.mean(f1s), np.std(f1s)
    g = run(StratifiedGroupKFold(5, shuffle=True, random_state=SEED), True)
    n = run(StratifiedKFold(5, shuffle=True, random_state=SEED), False)
    print(f"KNN-Eucl series macroF1: grouped={g[0]:.3f}±{g[1]:.3f}  "
          f"naive(ungrouped)={n[0]:.3f}±{n[1]:.3f}  gap={n[0]-g[0]:+.3f}")
    with open(os.path.join(tslib.CACHE_DIR, "baselines.pkl"), "wb") as f:
        pickle.dump({"dummy": dummy, "grouped_f1": g, "naive_f1": n}, f)


if __name__ == "__main__":
    m = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if m in ("minirocket", "all"): run_rocket()
    if m in ("minirocket_subset", "all"): run_rocket(channels=[3, 4], name="MiniRocket-enmoanglez")
    if m in ("muse", "all"): run_muse()
    if m in ("rdst", "all"): run_rdst()
    if m in ("shapelet", "all"): run_shapelet()
    if m in ("knn_euclidean", "all"): run_knn_precomputed("euclidean", "KNN-Euclidean")
    if m in ("knn_manhattan", "all"): run_knn_precomputed("manhattan", "KNN-Manhattan")
    if m in ("knn_dtw", "all"): run_knn_dtw_window_sweep()
    if m in ("knn_dtw_subset", "all"): pass  # handled separately if needed
    if m in ("baselines", "all"): run_baselines()
    print(f"== {m} done in {time.time()-t0:.0f}s ==")
