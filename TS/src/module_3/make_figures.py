"""Consolidate cached CVResults into comparison artifacts: comparison table
(printed + LaTeX), subject-level ROC/PR overlays, confusion-matrix grid, grouped
metric bar chart, and the DTW window-sweep curve."""
import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix,
                             roc_auc_score)

warnings.filterwarnings("ignore")
import tslib

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# display order + pretty names
ORDER = [
    ("KNN-Euclidean", "KNN (Euclidean)"),
    ("KNN-Manhattan", "KNN (Manhattan)"),
    ("KNN-DTW", "KNN (DTW)"),
    ("Shapelet-RF", "Shapelet Transform + RF"),
    ("RDST", "RDST (shapelet)"),
    ("MiniRocket", "MiniRocket"),
    ("MUSE", "MUSE"),
]


def load(name):
    p = os.path.join(tslib.CACHE_DIR, f"res_{name}.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    results = [(pretty, load(key)) for key, pretty in ORDER]
    results = [(p, r) for p, r in results if r is not None]
    baselines = None
    bp = os.path.join(tslib.CACHE_DIR, "baselines.pkl")
    if os.path.exists(bp):
        with open(bp, "rb") as f:
            baselines = pickle.load(f)

    # ---------- comparison table ----------
    rows = []
    for pretty, r in results:
        s = r.summary()
        rows.append((pretty, s["cv_score"], s["subj_f1"], s["subj_f1_std"],
                     s["subj_roc_auc"], s["subj_bal_acc"], s["ser_f1"]))
    cols = ["Model", "CV F1", "Subj F1", "±", "Subj AUC", "Subj balAcc", "Ser F1"]
    print("\n=== COMPARISON (subject-level headline) ===")
    print(f"{'Model':24s} {'CVf1':>5s} {'sF1':>5s} {'±':>5s} {'sAUC':>5s} "
          f"{'sBal':>5s} {'serF1':>5s}")
    for pretty, cv, f1, sd, auc, bal, ser in rows:
        cvs = f"{cv:.3f}" if not np.isnan(cv) else "  -- "
        print(f"{pretty:24s} {cvs:>5s} {f1:5.3f} {sd:5.3f} {auc:5.3f} "
              f"{bal:5.3f} {ser:5.3f}")
    if baselines:
        d = baselines["dummy"]
        print(f"{'Dummy (majority)':24s} {'  --':>5s} {d['subj_f1']:5.3f} "
              f"{'  --':>5s} {'0.500':>5s} {d['subj_bal_acc']:5.3f} {'  --':>5s}")
        g, n = baselines["grouped_f1"], baselines["naive_f1"]
        print(f"\nLeakage exhibit (KNN-Eucl, series macro-F1): "
              f"grouped={g[0]:.3f}±{g[1]:.3f}  naive={n[0]:.3f}±{n[1]:.3f}  "
              f"gap={n[0]-g[0]:+.3f}")

    # ---------- LaTeX table ----------
    best_f1 = max(r[2] for r in rows)
    best_auc = max(r[4] for r in rows)
    best_bal = max(r[5] for r in rows)
    best_cv = max((r[1] for r in rows if not np.isnan(r[1])), default=np.nan)
    lat = [r"\begin{table}[H]", r"\centering", r"\small",
           r"\caption{Time-series classifiers on the day-capped set, subject-grouped "
           r"nested CV. Subject-level metrics are the headline (mean over 5 outer "
           r"folds); CV F1 is the mean inner macro-F1 of the selected models.}",
           r"\label{tab:ts_clf}",
           r"\begin{tabular}{l c c c c}", r"\hline",
           r"Model & CV F1 & Subj F1 & Subj AUC & Subj bal-acc \\", r"\hline"]
    def b(v, best, fmt="{:.3f}"):
        s = fmt.format(v)
        return rf"\textbf{{{s}}}" if abs(v - best) < 1e-9 else s
    for pretty, cv, f1, sd, auc, bal, ser in rows:
        cvs = b(cv, best_cv) if not np.isnan(cv) else "--"
        lat.append(f"{pretty} & {cvs} & {b(f1,best_f1)} & {b(auc,best_auc)} & "
                   f"{b(bal,best_bal)} \\\\")
    if baselines:
        d = baselines["dummy"]
        lat.append(rf"Dummy (majority) & -- & {d['subj_f1']:.3f} & 0.500 & "
                   rf"{d['subj_bal_acc']:.3f} \\")
    lat += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(f"{tslib.FIG_DIR}/table_ts_clf.tex", "w") as f:
        f.write("\n".join(lat))

    # params table
    lat2 = [r"\begin{table}[H]", r"\centering", r"\small",
            r"\caption{Best configuration per method (subject-grouped CV).}",
            r"\label{tab:ts_params}", r"\begin{tabular}{l l}", r"\hline",
            r"Model & Best configuration \\", r"\hline"]
    for pretty, r in results:
        from collections import Counter
        cfgs = [tuple(sorted(d.items())) for d in r.best_params]
        common = Counter(cfgs).most_common(1)[0][0]
        cfg = ", ".join(f"{k}={v}" for k, v in common).replace("_", r"\_")
        if pretty == "MUSE":
            cfg = r"defaults (window\_inc=4, no bigrams/diff)"
        cfg = cfg or "defaults"
        lat2.append(f"{pretty} & {cfg} \\\\")
    lat2 += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(f"{tslib.FIG_DIR}/table_ts_params.tex", "w") as f:
        f.write("\n".join(lat2))

    # ---------- ROC + PR (subject level, pooled OOF) ----------
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(results)))
    for (pretty, r), col in zip(results, cmap):
        y, p = r.oof_subject["y"], r.oof_subject["p"]
        auc = roc_auc_score(y, p)
        fpr, tpr, _ = roc_curve(y, p)
        ax[0].plot(fpr, tpr, color=col, lw=1.6, label=f"{pretty} ({auc:.2f})")
        prec, rec, _ = precision_recall_curve(y, p)
        ax[1].plot(rec, prec, color=col, lw=1.6, label=pretty)
    ax[0].plot([0, 1], [0, 1], "k--", lw=.8)
    ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].set_title("Subject-level ROC")
    ax[0].legend(fontsize=7, loc="lower right")
    base = r.oof_subject["y"].mean()
    ax[1].axhline(base, color="k", ls="--", lw=.8)
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Subject-level PR")
    ax[1].legend(fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{tslib.FIG_DIR}/fig_roc_pr.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- confusion matrices grid ----------
    ncol = 4; nrow = int(np.ceil(len(results) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3 * ncol, 2.7 * nrow))
    for ax, (pretty, r) in zip(axes.ravel(), results):
        y, p = r.oof_subject["y"], r.oof_subject["p"]
        cm = confusion_matrix(y, (p >= .5).astype(int))
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(pretty, fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xlabel("pred"); ax.set_ylabel("true")
    for ax in axes.ravel()[len(results):]:
        ax.axis("off")
    fig.suptitle("Subject-level confusion matrices (pooled OOF)")
    fig.tight_layout(); fig.savefig(f"{tslib.FIG_DIR}/fig_confusion.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- grouped metric bar chart (mean±std over folds) ----------
    fig, ax = plt.subplots(figsize=(8.5, 4))
    metrics = [("subj_f1", "Subj F1"), ("subj_roc_auc", "Subj AUC"),
               ("subj_bal_acc", "Subj bal-acc")]
    x = np.arange(len(results)); w = 0.26
    for i, (key, lab) in enumerate(metrics):
        vals = [r.summary()[key] for _, r in results]
        errs = [r.summary().get(key + "_std", 0) for _, r in results]
        ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=2, label=lab)
    ax.axhline(0.5, color="k", ls="--", lw=.7)
    ax.set_xticks(x); ax.set_xticklabels([p for p, _ in results], rotation=30, ha="right")
    ax.set_ylabel("score"); ax.legend(); ax.set_title("Subject-level metrics (mean ± std over folds)")
    fig.tight_layout(); fig.savefig(f"{tslib.FIG_DIR}/fig_compare_bars.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- DTW window sweep ----------
    wc = os.path.join(tslib.CACHE_DIR, "dtw_window_curve.pkl")
    if os.path.exists(wc):
        with open(wc, "rb") as f:
            curve = pickle.load(f)
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        ws = sorted(curve); ax.plot(ws, [curve[w] for w in ws], "o-")
        ax.set_xlabel("Sakoe-Chiba window (frac of length)")
        ax.set_ylabel("subject F1"); ax.set_title("KNN-DTW band-width sweep")
        fig.tight_layout(); fig.savefig(f"{tslib.FIG_DIR}/fig_dtw_window.png", bbox_inches="tight")
        plt.close(fig)

    # subset ablation print
    sub = load("MiniRocket-enmoanglez")
    if sub is not None:
        s = sub.summary()
        full = load("MiniRocket").summary()
        print(f"\nMiniRocket channel ablation: all-5 AUC={full['subj_roc_auc']:.3f} "
              f"F1={full['subj_f1']:.3f}  |  enmo+anglez AUC={s['subj_roc_auc']:.3f} "
              f"F1={s['subj_f1']:.3f}")
    print("\nfigures + tables written to", tslib.FIG_DIR)


if __name__ == "__main__":
    main()
