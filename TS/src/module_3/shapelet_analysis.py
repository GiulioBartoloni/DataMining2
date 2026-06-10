"""Retrieve and PLOT the most discriminative shapelets (the analysis deliverable).

Fits a RandomShapeletTransform on a single subject-grouped training split, ranks
shapelets by information gain, plots the top ones (by channel/class), overlays the
best shapelet on a matching example series, and runs the subject-fingerprint guard
(are the top shapelets dominated by one prolific subject?).
"""
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
import tslib
from sklearn.model_selection import StratifiedGroupKFold
from aeon.transformations.collection.shapelet_based import RandomShapeletTransform

plt.rcParams.update({"figure.dpi": 120, "font.size": 10})
C0, C1 = "#4C72B0", "#C44E52"
# RandomShapeletTransform shapelet tuple layout:
# (info_gain, length, position, channel, dilation, series_index, data_array)
IG, LEN, POS, CH, DIL, SIDX, ARR = range(7)


def main():
    data = tslib.load_raw()
    X, y, groups = tslib.build_arrays(data)
    Xc, yc, gc, nd = tslib.day_cap(X, y, groups, k=10)
    st = tslib.channel_scaler(Xc)
    Xs = tslib.apply_scaler(Xc, st)

    # one grouped split -> train shapelets on the train side
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=tslib.SEED)
    tr, te = next(outer.split(Xs, yc, gc))
    rst = RandomShapeletTransform(n_shapelet_samples=4000, max_shapelets=80,
                                  random_state=tslib.SEED, n_jobs=4)
    rst.fit(Xs[tr], yc[tr])
    shp = sorted(rst.shapelets, key=lambda s: -s[IG])
    print(f"retrieved {len(shp)} shapelets; top info-gain={shp[0][IG]:.3f}")

    # --- subject-fingerprint guard ---
    train_groups = gc[tr]
    top_subjects = [int(train_groups[s[SIDX]]) for s in shp[:20]]
    uniq, cnt = np.unique(top_subjects, return_counts=True)
    print(f"top-20 shapelets come from {len(uniq)} distinct subjects; "
          f"max from one subject = {cnt.max()} "
          f"(fingerprint risk {'LOW' if cnt.max() <= 4 else 'CHECK'})")

    # --- Fig: top-6 shapelets, colored by the class of their source series ---
    fig, axes = plt.subplots(2, 3, figsize=(11, 5))
    for ax, s in zip(axes.ravel(), shp[:6]):
        arr = np.asarray(s[ARR]).ravel()
        cls = int(yc[tr][s[SIDX]])
        ax.plot(arr, color=(C0 if cls == 0 else C1), lw=1.6)
        ax.set_title(f"{tslib.CHANNELS[s[CH]]} | IG={s[IG]:.3f}\n"
                     f"len={s[LEN]} class={cls}", fontsize=9)
        ax.grid(alpha=.3)
    fig.suptitle("Top-6 discriminative shapelets (color = source-series class)")
    fig.tight_layout(); fig.savefig(f"{tslib.FIG_DIR}/fig_shapelets_top.png",
                                    bbox_inches="tight")
    plt.close(fig)

    # --- Fig: best shapelet localized on its source series ---
    best = shp[0]
    arr = np.asarray(best[ARR]).ravel()
    series = Xs[tr][best[SIDX], best[CH]]
    pos, L, dil = best[POS], best[LEN], best[DIL]
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.plot(series, color="#999", lw=1, label=f"series (subj {int(gc[tr][best[SIDX]])}, "
            f"class {int(yc[tr][best[SIDX]])})")
    locs = pos + np.arange(L) * dil
    locs = locs[locs < len(series)]
    ax.plot(locs, series[locs], color="#C44E52", lw=2.4,
            label=f"best shapelet ({tslib.CHANNELS[best[CH]]}, IG={best[IG]:.3f})")
    ax.set_title("Best shapelet located on its source series")
    ax.set_xlabel("timestep (scaled units)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{tslib.FIG_DIR}/fig_shapelet_localized.png",
                                    bbox_inches="tight")
    plt.close(fig)

    # channel usage of top-20
    chs = [tslib.CHANNELS[s[CH]] for s in shp[:20]]
    u, c = np.unique(chs, return_counts=True)
    print("top-20 shapelet channel usage:", dict(zip(u.tolist(), c.tolist())))
    print("shapelet figures written.")


if __name__ == "__main__":
    main()
