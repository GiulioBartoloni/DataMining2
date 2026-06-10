"""Precompute full pairwise DTW distance matrices on the day-capped, globally
per-channel-scaled set, once per Sakoe-Chiba window. Cached to cache/dtw_w*.npy.

This is the DTW cost guard: once the matrix exists, KNN over any fold / any k is
just submatrix indexing (sklearn metric='precomputed').
"""
import os
import time

import numpy as np
from aeon.distances import dtw_pairwise_distance

import tslib

WINDOWS = [0.02, 0.05, 0.10]


def main():
    data = tslib.load_raw()
    X, y, groups = tslib.build_arrays(data)
    Xc, yc, gc, nd = tslib.day_cap(X, y, groups, k=10)
    # global per-channel scaling (label-independent) so the cached matrix is valid
    Xs = tslib.apply_scaler(Xc, tslib.channel_scaler(Xc))
    np.save(os.path.join(tslib.CACHE_DIR, "capped_y.npy"), yc)
    np.save(os.path.join(tslib.CACHE_DIR, "capped_groups.npy"), gc)
    np.save(os.path.join(tslib.CACHE_DIR, "capped_ndays.npy"), nd)
    print(f"capped set: {Xs.shape}")

    for w in WINDOWS:
        out = os.path.join(tslib.CACHE_DIR, f"dtw_w{w}.npy")
        if os.path.exists(out):
            print(f"  window={w}: cached, skip")
            continue
        t = time.time()
        D = dtw_pairwise_distance(Xs, window=w)
        np.save(out, D.astype(np.float32))
        print(f"  window={w}: {D.shape} in {time.time()-t:.0f}s -> {out}")

    # also Manhattan & Euclidean pairwise (cheap, same scaled set) for KNN parity
    from aeon.distances import (
        manhattan_pairwise_distance,
        euclidean_pairwise_distance,
    )
    for name, fn in [("euclidean", euclidean_pairwise_distance),
                     ("manhattan", manhattan_pairwise_distance)]:
        out = os.path.join(tslib.CACHE_DIR, f"{name}.npy")
        if os.path.exists(out):
            print(f"  {name}: cached, skip"); continue
        t = time.time()
        D = fn(Xs)
        np.save(out, D.astype(np.float32))
        print(f"  {name}: {D.shape} in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main()
