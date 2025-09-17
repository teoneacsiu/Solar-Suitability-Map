# scripts/apply_hard_filters.py
# ==========================================================
# Scop:
#  - Aplică "excluderi HARD" peste datele procesate, FĂRĂ să reproceseze România.
#  - Folosește rasterele existente din data/processed:
#       - sentinel_composite.tif (benzi S2: B2,B4,B8,B11)
#       - slope.tif (pantă în grade)
#       - ghi.tif   (Global Horizontal Irradiance - opțional pentru raport)
#       - (opțional) suitability_map.tif (harta modelului; dacă există, creează și versiunea filtrată)
#
# Excluderi implementate:
#   1) Panta > --slope-max (default 7°)
#   2) Pădure: NDVI > --ndvi-forest (default 0.55)
#   3) Urban:  NDBI > --ndbi-urban (default 0.05)
#   4) Apă:    NDWI_Gao > --ndwi-water (default 0.10), cu buffer --buffer-water-m (default 200 m)
#
# Ieșiri:
#   data/processed/masks/  (măști intermediare binare 0/1)
#     - slope_excl.tif
#     - forest_excl.tif
#     - urban_excl.tif
#     - water_excl.tif
#     - hard_invalid_mask.tif  (OR pe toate excluderile)
#   data/processed/suitability_masked.tif  (dacă există suitability_map.tif)
#
# Notă:
#  - NDVI = (B8 - B4) / (B8 + B4)
#  - NDBI = (B11 - B8) / (B11 + B8)
#  - NDWI_Gao = (B8 - B11) / (B8 + B11)  (proxy pentru apă fără B3; funcționează rezonabil)
#  - Bufferul se face cu dilatare morfologică în pixeli (calculăm pixeli din metri).
# ==========================================================

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine
from scipy.ndimage import binary_dilation

from config import PROC_DIR

PROC_DIR = Path(PROC_DIR)
MASKS_DIR = PROC_DIR / "masks"
MASKS_DIR.mkdir(parents=True, exist_ok=True)

def _read_s2_bands(path):
    """Citește benzile S2 dintr-un GeoTIFF 4 benzi: B2,B4,B8,B11 (în această ordine)."""
    with rasterio.open(path) as src:
        b2, b4, b8, b11 = src.read().astype("float32")
        meta = src.meta.copy()
        transform = src.transform
        crs = src.crs
        # mărimea pixelului (m) – presupunem proiecție metrică (ex. EPSG:3857)
        px_x = abs(transform.a)
        px_y = abs(transform.e)
    return (b2, b4, b8, b11), meta, transform, crs, px_x, px_y

def _save_mask_like(ref_meta, out_path, mask_uint8):
    """Salvează un raster mask (0/1) cu profil compatibil cu meta de referință."""
    meta = ref_meta.copy()
    meta.update(
        dtype="uint8",
        count=1,
        nodata=0,
        compress="lzw",
        tiled=True,
        bigtiff="IF_SAFER",
    )
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(mask_uint8, 1)

def _save_float_like(ref_meta, out_path, arr_float):
    """Salvează un raster float32 (ex. suitability filtrată)."""
    meta = ref_meta.copy()
    meta.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
        compress="lzw",
        tiled=True,
        bigtiff="IF_SAFER",
    )
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr_float.astype("float32"), 1)

def main():
    p = argparse.ArgumentParser(description="Aplică excluderi HARD peste rasterele procesate")
    # praguri
    p.add_argument("--slope-max", type=float, default=7.0, help="Exclude pante > acest prag (grade)")
    p.add_argument("--ndvi-forest", type=float, default=0.55, help="Exclude NDVI > prag (proxy pădure)")
    p.add_argument("--ndbi-urban", type=float, default=0.05, help="Exclude NDBI > prag (proxy urban)")
    p.add_argument("--ndwi-water", type=float, default=0.10, help="Exclude NDWI_Gao > prag (proxy apă)")
    # buffere (m)
    p.add_argument("--buffer-water-m", type=float, default=200.0, help="Buffer pentru apă (metri)")
    p.add_argument("--buffer-urban-m", type=float, default=200.0, help="Buffer pentru urban (metri)")
    # fișiere
    p.add_argument("--s2", type=str, default=str(PROC_DIR / "sentinel_composite.tif"),
                   help="Calea la sentinel_composite.tif (B2,B4,B8,B11)")
    p.add_argument("--slope", type=str, default=str(PROC_DIR / "slope.tif"),
                   help="Calea la slope.tif (grade)")
    p.add_argument("--in-prob", type=str, default=str(PROC_DIR / "suitability_map.tif"),
                   help="Hartă probabilități (dacă există, scriem și suitability_masked.tif)")
    args = p.parse_args()

    # 1) Citește benzile S2 și metadatele
    s2_bands, s2_meta, s2_transform, s2_crs, px_x, px_y = _read_s2_bands(args.s2)
    B2, B4, B8, B11 = s2_bands
    H, W = B2.shape

    # 2) NDVI / NDBI / NDWI (Gao) – toate pe float32, cu protecție la împărțire
    eps = 1e-6
    ndvi = (B8 - B4) / (B8 + B4 + eps)
    ndbi = (B11 - B8) / (B11 + B8 + eps)
    ndwi = (B8 - B11) / (B8 + B11 + eps)  # Gao 1996

    # 3) Citește panta (re-eșantionată deja la același grid în preprocess)
    with rasterio.open(args.slope) as src_s:
        slope = src_s.read(1).astype("float32")

    # 4) Construiește excluderi binare (0/1)
    slope_excl = (slope > args.slope_max) & np.isfinite(slope)
    forest_excl = (ndvi > args.ndvi_forest) & np.isfinite(ndvi)
    urban_excl = (ndbi > args.ndbi_urban) & np.isfinite(ndbi)
    water_raw = (ndwi > args.ndwi_water) & np.isfinite(ndwi)

    # 5) Buffer-e în pixeli (din metri)
    #    Notă: dacă pixelul tău e 200 m, un buffer de 200 m = 1 pixel
    px_size = float(max(px_x, px_y))
    buf_water_px = int(np.ceil(args.buffer_water_m / (px_size + eps)))
    buf_urban_px = int(np.ceil(args.buffer_urban_m / (px_size + eps)))

    if buf_water_px > 0:
        water_excl = binary_dilation(water_raw, iterations=buf_water_px)
    else:
        water_excl = water_raw

    if buf_urban_px > 0:
        urban_excl = binary_dilation(urban_excl, iterations=buf_urban_px)

    # 6) Mască HARD finală = OR pe toate excluderile (1 = INVALID / exclus)
    hard_invalid = slope_excl | forest_excl | urban_excl | water_excl

    # 7) Salvează măștile individuale + masca finală
    _save_mask_like(s2_meta, MASKS_DIR / "slope_excl.tif", slope_excl.astype("uint8"))
    _save_mask_like(s2_meta, MASKS_DIR / "forest_excl.tif", forest_excl.astype("uint8"))
    _save_mask_like(s2_meta, MASKS_DIR / "urban_excl.tif", urban_excl.astype("uint8"))
    _save_mask_like(s2_meta, MASKS_DIR / "water_excl.tif", water_excl.astype("uint8"))
    _save_mask_like(s2_meta, MASKS_DIR / "hard_invalid_mask.tif", hard_invalid.astype("uint8"))

    print("✔ Măști HARD scrise în:", MASKS_DIR)

    # 8) Dacă avem deja harta de probabilități, scriem varianta “mascată”
    prob_path = Path(args.in_prob)
    if prob_path.exists():
        with rasterio.open(prob_path) as src_p:
            prob = src_p.read(1).astype("float32")
            pmeta = src_p.meta.copy()
        prob_masked = prob.copy()
        prob_masked[hard_invalid] = np.nan  # invalid → NaN (transparență în vizualizare)
        _save_float_like(pmeta, PROC_DIR / "suitability_masked.tif", prob_masked)
        print("✔ suitability_masked.tif scris (probabilitate + excluderi HARD)")
    else:
        print("ℹ Nu am găsit", prob_path.name, "— am generat doar măștile HARD.")

    # 9) Mic raport de acoperire
    total = H * W
    excl = int(hard_invalid.sum())
    print(f"ℹ Pixeli excluși (HARD): {excl} din {total}  (~{100*excl/total:.1f}%)")
    print("   breakdown aproximativ (suprapuneri posibile):")
    for name, m in [("Pantă", slope_excl), ("Pădure", forest_excl),
                    ("Urban", urban_excl), ("Apă+buffer", water_excl)]:
        print(f"   - {name:12s}: {int(m.sum())}")

if __name__ == "__main__":
    main()
