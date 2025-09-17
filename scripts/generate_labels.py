# scripts/generate_labels.py
import os
import argparse
import numpy as np
import pandas as pd
import rasterio

from config import PROC_DIR

def main():
    parser = argparse.ArgumentParser(description="Genereaza labels.csv automat din rasterele procesate")
    parser.add_argument("--n_pos", type=int, default=300, help="numar tinte pozitive")
    parser.add_argument("--n_neg", type=int, default=300, help="numar tinte negative")
    parser.add_argument("--ghi_q", type=float, default=0.70, help="cuantila GHI pt a considera 'ridicat' (0..1)")
    parser.add_argument("--ndvi_lo", type=float, default=0.10, help="limita inferioara NDVI pentru pozitiv")
    parser.add_argument("--ndvi_hi", type=float, default=0.40, help="limita superioara NDVI pentru pozitiv")
    parser.add_argument("--slope_pos", type=float, default=5.0, help="panta maxima pt pozitiv (grade)")
    parser.add_argument("--slope_neg", type=float, default=10.0, help="panta minima pt negativ (grade)")
    parser.add_argument("--ndvi_forest", type=float, default=0.45, help="NDVI peste care consideram padure densa")
    args = parser.parse_args()

    # cai
    s2_path    = os.path.join(PROC_DIR, "sentinel_composite.tif")  # B2,B4,B8,B11
    slope_path = os.path.join(PROC_DIR, "slope.tif")
    ghi_path   = os.path.join(PROC_DIR, "ghi.tif")

    # citire benzi
    with rasterio.open(s2_path) as src:
        B2, B4, B8, B11 = src.read().astype("float32")
        height, width = src.height, src.width

    with rasterio.open(slope_path) as src_s:
        slope = src_s.read(1).astype("float32")

    with rasterio.open(ghi_path) as src_g:
        ghi = src_g.read(1).astype("float32")

    # calcule NDVI
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)

    # masca pixeli valizi (fara NaN/inf)
    valid = np.isfinite(ndvi) & np.isfinite(slope) & np.isfinite(ghi)

    # prag GHI ridicat = cuantila pe pixeli valizi
    ghi_thr = np.quantile(ghi[valid], args.ghi_q)

    # reguli pozitive (toate simultan)
    pos_mask = (
        valid &
        (slope < args.slope_pos) &
        (ndvi >= args.ndvi_lo) & (ndvi <= args.ndvi_hi) &
        (ghi >= ghi_thr)
    )

    # reguli negative (oricare)
    neg_mask_hard = (
        valid &
        (
            (slope > args.slope_neg) |
            (ndvi < 0.0) |
            (ndvi > args.ndvi_forest)
        )
    )

    # daca nu avem destui pozitivi, relaxeaza usor GHI sau NDVI
    pos_idx = np.argwhere(pos_mask)
    if pos_idx.shape[0] < args.n_pos:
        # relaxare: coboara putin pragul GHI la cuantila 0.6
        ghi_thr_relax = np.quantile(ghi[valid], 0.60)
        pos_mask_relax = (
            valid &
            (slope < args.slope_pos) &
            (ndvi >= (args.ndvi_lo - 0.02)) & (ndvi <= (args.ndvi_hi + 0.02)) &
            (ghi >= ghi_thr_relax)
        )
        pos_idx = np.argwhere(pos_mask_relax)

    # daca nu avem destui negativi duri, completeaza cu negativi generali (restul valid - pozitiv)
    neg_idx = np.argwhere(neg_mask_hard)
    if neg_idx.shape[0] < args.n_neg:
        neg_mask_soft = valid & (~pos_mask)
        neg_idx = np.argwhere(neg_mask_soft)

    # esantionare aleatoare fara inlocuire
    rng = np.random.default_rng(42)
    if pos_idx.shape[0] == 0:
        raise SystemExit("Nu exista pixeli pozitivi dupa reguli. Verifica AOI sau praguri.")
    if neg_idx.shape[0] == 0:
        raise SystemExit("Nu exista pixeli negativi dupa reguli. Verifica AOI sau praguri.")

    pos_pick = pos_idx[rng.choice(pos_idx.shape[0], size=min(args.n_pos, pos_idx.shape[0]), replace=False)]
    neg_pick = neg_idx[rng.choice(neg_idx.shape[0], size=min(args.n_neg, neg_idx.shape[0]), replace=False)]

    # contruieste DataFrame row,col,label
    pos_df = pd.DataFrame({"row": pos_pick[:,0], "col": pos_pick[:,1], "label": 1})
    neg_df = pd.DataFrame({"row": neg_pick[:,0], "col": neg_pick[:,1], "label": 0})
    df = pd.concat([pos_df, neg_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle

    # salveaza
    out_csv = os.path.join(PROC_DIR, "labels.csv")
    os.makedirs(PROC_DIR, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"✔ labels.csv scris: {out_csv}")
    print(f"   Pozitive: {pos_df.shape[0]}, Negative: {neg_df.shape[0]}")
    print(f"   Dimensiune raster: {height} x {width}")
    print(f"   Prag GHI (q={args.ghi_q:.2f}): {ghi_thr:.3f}")

if __name__ == "__main__":
    main()
