# scripts/make_labels.py
# --------------------------------------------------------------
# Creează un set de evaluare aliniat pe grila hărții tale:
#  - Pozitive: situri PV OSM (poligoane/ puncte bufferizate)
#  - Negative: apă/urban/pădure/pantă mare (din s2 + slope procesate)
#  - Eșantionează aleator N pozitive și N negative → eval_points.npz
# NOTĂ: folosește strict datele procesate deja (fără re-procesări).
# --------------------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.warp import reproject, Resampling
from shapely.geometry import Point
from shapely.ops import transform as shp_transform
import pyproj

from rasterio.warp import transform_bounds

def reproject_raster_to_ref(src_arr, src_transform, src_crs, ref_transform, ref_crs, ref_shape):
    dst = np.full(ref_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_arr.astype(np.float32),
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.bilinear,
    )
    return dst

def compute_indices_from_s2(s2_path):
    with rasterio.open(s2_path) as src:
        B2, B4, B8, B11 = src.read().astype("float32")
        tr, crs = src.transform, src.crs
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    ndbi = (B11 - B8) / (B11 + B8 + 1e-6)
    ndwi = (B2 - B11) / (B2 + B11 + 1e-6)
    return dict(ndvi=ndvi, ndbi=ndbi, ndwi=ndwi, transform=tr, crs=crs)

def main():
    p = argparse.ArgumentParser(description="Build evaluation points from OSM PV sites + negatives.")
    p.add_argument("--sites", required=True, help="GeoJSON with OSM solar sites (from fetch_osm_solar.py)")
    p.add_argument("--map",   required=True, help="GeoTIFF used as reference grid (e.g., data/processed/suitability_adjusted.tif)")
    p.add_argument("--s2",    required=True, help="Processed Sentinel-2 composite (B2,B4,B8,B11) path")
    p.add_argument("--slope", required=True, help="Processed slope raster path")

    # praguri pentru negative (aceleași cu vizualizarea ta)
    p.add_argument("--slope-max",   type=float, default=7.0)
    p.add_argument("--ndvi-forest", type=float, default=0.55)
    p.add_argument("--ndwi-water",  type=float, default=0.05)
    p.add_argument("--ndbi-urban",  type=float, default=0.05)

    p.add_argument("--point-buffer-m", type=float, default=150.0, help="Buffer (m) pentru puncte OSM ca să devină arii")
    p.add_argument("--samples-per-class", type=int, default=10000, help="Câte puncte pozitive/negative să extragi")
    p.add_argument("--out", default="data/labels/eval_points.npz")

    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # grila de referință = harta ta
    with rasterio.open(args.map) as ref:
        H, W = ref.height, ref.width
        ref_tr, ref_crs = ref.transform, ref.crs

    # 1) Rasterizare pozitive
    gdf = gpd.read_file(args.sites).to_crs(ref_crs)

    # Buffer pentru puncte (în metri). Dacă CRS-ul e geografic, buffer în 3857 apoi înapoi.
    if not ref_crs or ref_crs.to_string().upper().startswith("EPSG:4326"):
        to3857 = pyproj.Transformer.from_crs(ref_crs, "EPSG:3857", always_xy=True).transform
        to_ref = pyproj.Transformer.from_crs("EPSG:3857", ref_crs, always_xy=True).transform
        geoms = []
        for g in gdf.geometry:
            if g.geom_type == "Point":
                g2 = shp_transform(to3857, g).buffer(args.point_buffer_m)
                geoms.append(shp_transform(to_ref, g2))
            else:
                geoms.append(g)
        gdf = gdf.set_geometry(geoms)
    else:
        # proiectat metric: buffer direct
        geoms = []
        for g in gdf.geometry:
            if g.geom_type == "Point":
                geoms.append(g.buffer(args.point_buffer_m))
            else:
                geoms.append(g)
        gdf = gdf.set_geometry(geoms)

    pos_mask = features.rasterize(
        [(geom, 1) for geom in gdf.geometry if geom and not geom.is_empty],
        out_shape=(H, W),
        transform=ref_tr,
        fill=0,
        all_touched=True
    ).astype(bool)

    # 2) Negative din condiții (reproiectate pe grila de referință)
    idx = compute_indices_from_s2(args.s2)
    with rasterio.open(args.slope) as ssrc:
        slope_raw = ssrc.read(1).astype("float32")
        slope_tr, slope_crs = ssrc.transform, ssrc.crs

    ndvi = reproject_raster_to_ref(idx["ndvi"], idx["transform"], idx["crs"], ref_tr, ref_crs, (H, W))
    ndbi = reproject_raster_to_ref(idx["ndbi"], idx["transform"], idx["crs"], ref_tr, ref_crs, (H, W))
    ndwi = reproject_raster_to_ref(idx["ndwi"], idx["transform"], idx["crs"], ref_tr, ref_crs, (H, W))
    slope= reproject_raster_to_ref(slope_raw, slope_tr, slope_crs, ref_tr, ref_crs, (H, W))

    water_mask  = ndwi  > float(args.ndwi_water)
    urban_mask  = ndbi  > float(args.ndbi_urban)
    forest_mask = ndvi  > float(args.ndvi_forest)
    slope_mask  = slope > float(args.slope_max)
    neg_mask = water_mask | urban_mask | forest_mask | slope_mask

    # 3) Eșantionare puncte
    rng = np.random.default_rng(42)

    pos_idx = np.argwhere(pos_mask)
    if len(pos_idx) == 0:
        raise SystemExit("No positive pixels found from OSM sites. Check your sites file.")
    pos_take = pos_idx[rng.choice(len(pos_idx), size=min(args.samples_per_class, len(pos_idx)), replace=False)]

    neg_idx = np.argwhere(neg_mask & ~pos_mask)
    if len(neg_idx) == 0:
        raise SystemExit("No negative pixels found. Check thresholds.")
    neg_take = neg_idx[rng.choice(len(neg_idx), size=min(args.samples_per_class, len(neg_idx)), replace=False)]

    rows = np.concatenate([pos_take[:,0], neg_take[:,0]])
    cols = np.concatenate([pos_take[:,1], neg_take[:,1]])
    y    = np.concatenate([np.ones(len(pos_take), dtype=np.uint8),
                           np.zeros(len(neg_take), dtype=np.uint8)])

    np.savez_compressed(args.out, rows=rows, cols=cols, y=y)
    print(f"✔ Saved eval set: {len(y)} pts → {args.out} "
          f"(pos {len(pos_take)}, neg {len(neg_take)})")

if __name__ == "__main__":
    main()
