"""EDA figures for the report: class balance (series vs subject), series-per-
subject distribution (leakage motivation), and example series per class."""
import warnings
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
import tslib

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
C0, C1 = "#4C72B0", "#C44E52"


def main():
    data = tslib.load_raw()
    X, y, groups = tslib.build_arrays(data)
    sids = np.unique(groups)
    sy = np.array([y[groups == s][0] for s in sids])
    counts = np.array([(groups == s).sum() for s in sids])

    # --- Fig 1: class balance series vs subject ---
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.2))
    for a, (lab, vals, tot) in zip(
        ax, [("Series-level", y, len(y)), ("Subject-level", sy, len(sy))]):
        n0, n1 = (vals == 0).sum(), (vals == 1).sum()
        a.bar(["non-prob. (0)", "problematic (1)"], [n0, n1], color=[C0, C1])
        for i, v in enumerate([n0, n1]):
            a.text(i, v, f"{v}\n{v/tot:.1%}", ha="center", va="bottom", fontsize=9)
        a.set_title(lab); a.set_ylim(0, max(n0, n1) * 1.18)
    fig.suptitle("Class distribution: series vs subject level")
    fig.tight_layout(); fig.savefig(f"{tslib.FIG_DIR}/fig_class_balance.png", bbox_inches="tight")
    plt.close(fig)

    # --- Fig 2: series-per-subject histogram (leakage motivation) ---
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.hist(counts, bins=range(1, counts.max() + 2), color="#55A868", edgecolor="white")
    ax.axvline(np.median(counts), color="k", ls="--", lw=1,
               label=f"median {np.median(counts):.0f}")
    ax.set_xlabel("series (days) per subject"); ax.set_ylabel("# subjects")
    ax.set_title(f"{len(sids)} subjects, {len(y)} series "
                 f"(max {counts.max()}/subject)")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{tslib.FIG_DIR}/fig_series_per_subject.png", bbox_inches="tight")
    plt.close(fig)

    # --- Fig 3: example series per class (mean enmo/anglez) ---
    fig, axes = plt.subplots(2, 5, figsize=(12, 4.2), sharex=True)
    rng = np.random.default_rng(0)
    for row, cls in enumerate([0, 1]):
        idx = rng.choice(np.where(y == cls)[0], 1)[0]
        for c, ch in enumerate(tslib.CHANNELS):
            axes[row, c].plot(X[idx, c], color=(C0 if cls == 0 else C1), lw=0.8)
            if row == 0:
                axes[row, c].set_title(ch)
            if c == 0:
                axes[row, c].set_ylabel(f"class {cls}")
    fig.suptitle("Example series (one subject-day per class)")
    fig.tight_layout(); fig.savefig(f"{tslib.FIG_DIR}/fig_example_series.png", bbox_inches="tight")
    plt.close(fig)

    print("EDA figures written. Stats:")
    print(f"  series={len(y)} subjects={len(sids)} "
          f"series_prev={y.mean():.3f} subject_prev={sy.mean():.3f}")
    print(f"  series/subject: min={counts.min()} max={counts.max()} "
          f"mean={counts.mean():.1f} median={np.median(counts):.1f}")
    # channel scale table
    for c, ch in enumerate(tslib.CHANNELS):
        v = X[:, c, :]
        print(f"  {ch:7s} min={v.min():.2f} max={v.max():.2f} "
              f"mean={v.mean():.3f} std={v.std():.3f}")


if __name__ == "__main__":
    main()
