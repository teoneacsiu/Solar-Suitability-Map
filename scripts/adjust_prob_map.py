# scripts/adjust_prob_map.py
# ==========================================================
# Scop:
#   Re-ponderare "SOFT" a probabilității modelului, FĂRĂ să tai pixeli.
#   Ideea: folosești reguli blânde (favorizezi NDVI mediu, penalizezi ușor pantă,
#         urban și apă) pentru a obține o hartă "suitability_adjusted.tif".
#
# Intrări (din data/processed):
#   - sentinel_composite.tif (B2,B4,B8,B11) → NDVI, NDBI, NDWI
#   - slope.tif
#   - suitability_map.tif   (harta de probabilitate originală)
#
# Ieșire:
#   - suitability_adjusted.tif
#
# Parametri principali:
#   --w-ndvi    greutatea preferinței pentru NDVI în [ndvi_lo, ndvi_hi] (default 0.3)
#   --w-slope   greutatea penalizării pantei (default 0.2)
#   --w-urban   greutatea penalizării NDBI (default 0.2)
#   --w-water   greutatea penalizării NDWI (default 0.2)
#   --w-ghi     greutatea bonusului relativ la GHI (opțional, default 0.1)
#
# Notă:
#   - Toate greutățile sunt în [0..1] și se aplică multiplicativ pe probabilități.
#   - NU setează NaN/0; pentru excluderi stricte folosește apply_hard_filters.py.
# ==========================================================

import argparse
from pathlib import Path

import numpy as np
import rasterio

from config import PROC_DIR

PROC_DIR = Path(PROC_DIR)

def _read_s2(path):
    with rasterio.open(path) as src:
        B2, B4, B8, B11 = src.read().astype("float32")
        meta = src.meta.copy()
    return (B2, B4, B8, B11), meta

def _save_float_like(ref_meta, out_path, arr_float):
    meta = ref_meta.copy()
    meta.update(dtype="float32", count=1, nodata=np.nan, compress="lzw", tiled=True, bigtiff="IF_SAFER")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr_float.astype("float32"), 1)

def main():
    p = argparse.ArgumentParser(description="Re-ponderare SOFT a probabilității")
    p.add_argument("--prob", type=str, default=str(PROC_DIR / "suitability_map.tif"))
    p.add_argument("--s2",   type=str, default=str(PROC_DIR / "sentinel_composite.tif"))
    p.add_argument("--slope", type=str, default=str(PROC_DIR / "slope.tif"))
    p.add_argument("--ghi", type=str, default=str(PROC_DIR / "ghi.tif"))
    # greutăți
    p.add_argument("--w-ndvi", type=float, default=0.30)
    p.add_argument("--w-slope", type=float, default=0.20)
    p.add_argument("--w-urban", type=float, default=0.20)
    p.add_argument("--w-water", type=float, default=0.20)
    p.add_argument("--w-ghi",   type=float, default=0.10)
    # preferințe
    p.add_argument("--ndvi-lo", type=float, default=0.10, help="NDVI preferat minim")
    p.add_argument("--ndvi-hi", type=float, default=0.40, help="NDVI preferat maxim")
    p.add_argument("--ndbi-urban", type=float, default=0.05, help="Urban proxy (mai mare = mai urban)")
    p.add_argument("--ndwi-water", type=float, default=0.10, help="Apă proxy (mai mare = mai acvatic)")
    p.add_argument("--slope-ref", type=float, default=7.0, help="Pantă de referință (grade)")
    args = p.parse_args()

    # Citește probabilitatea de bază
    with rasterio.open(args.prob) as src_p:
        prob = src_p.read(1).astype("float32")
        pmeta = src_p.meta.copy()

    # Citește S2 și calculează indici
    (B2, B4, B8, B11), s2meta = _read_s2(args.s2)
    eps = 1e-6
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    ndbi = (B11 - B8) / (B11 + B8 + eps)
    ndwi = (B8 - B11) / (B8 + B11 + eps)

    # Slope
    with rasterio.open(args.slope) as src_s:
        slope = src_s.read(1).astype("float32")

    # GHI (opțional)
    try:
        with rasterio.open(args.ghi) as src_g:
            ghi = src_g.read(1).astype("float32")
    except:
        ghi = None

    # 1) Greutate NDVI – preferăm intervalul [ndvi_lo, ndvi_hi].
    #    Implementare: "tent" (triunghi) – 1.0 în centru și scade spre 0 în afara intervalului.
    lo, hi = args.ndvi_lo, args.ndvi_hi
    mid = 0.5 * (lo + hi)
    width = (hi - lo) / 2.0 + eps
    w_ndvi_shape = 1.0 - np.minimum(1.0, np.abs(ndvi - mid) / width)  # [0..1]
    w_ndvi = 1.0 - args.w_ndvi + args.w_ndvi * w_ndvi_shape          # mix spre [1-w, 1]

    # 2) Greutate pantă – penalizare ușoară peste slope-ref (exp decădere)
    slope_norm = np.clip(slope / (args.slope_ref + eps), 0, 5)
    w_slope_shape = np.exp(-1.0 * slope_norm)  # 1.0 la pantă mică; scade exponențial
    w_slope = 1.0 - args.w_slope + args.w_slope * w_slope_shape

    # 3) Greutate urban – NDBI mare => penalizare
    #    Normalizăm în [0..1] în jurul pragului (soft).
    ndbi_norm = np.clip((ndbi - args.ndbi_urban) / 0.2, 0, 1)  # 0 sub prag; 1 mult peste
    w_urban = 1.0 - args.w_urban * ndbi_norm

    # 4) Greutate apă – NDWI mare => penalizare
    ndwi_norm = np.clip((ndwi - args.ndwi_water) / 0.2, 0, 1)
    w_water = 1.0 - args.w_water * ndwi_norm

    # 5) Greutate GHI – bonus relativ: normalizăm GHI la [0..1] și aplicăm un mic boost
    if ghi is not None and np.isfinite(ghi).any():
        gmin = np.nanpercentile(ghi, 5)
        gmax = np.nanpercentile(ghi, 95)
        gnorm = np.clip((ghi - gmin) / (gmax - gmin + eps), 0, 1)
        w_ghi = 1.0 - args.w_ghi + args.w_ghi * gnorm
    else:
        w_ghi = 1.0

    # Combinație multiplicativă – păstrăm NaN-urile inițiale
    valid = np.isfinite(prob)
    adjusted = prob.copy()
    weight = (w_ndvi * w_slope * w_urban * w_water * w_ghi).astype("float32")
    adjusted[valid] = np.clip(prob[valid] * weight[valid], 0.0, 1.0)

    out_path = PROC_DIR / "suitability_adjusted.tif"
    _save_float_like(pmeta, out_path, adjusted)
    print("✔ suitability_adjusted.tif scris (probabilitate re-ponderată soft)")

if __name__ == "__main__":
    main()
