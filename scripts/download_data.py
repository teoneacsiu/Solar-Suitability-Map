# scripts/download_data.py
import argparse
from pathlib import Path
import sys
import traceback

import ee
import geemap

from config import (
    AOI, START_DATE, END_DATE, CLOUD_PCT,
    RAW_DIR, EE_PROJECT, EXPORT_SCALE
)

RAW_DIR = Path(RAW_DIR)

def init_ee():
    try:
        ee.Initialize(project=EE_PROJECT)
    except Exception:
        # Fallback to default init if project-specific init fails
        ee.Initialize()

def rect_from_bbox(b):
    # b = [xmin, ymin, xmax, ymax] in lon/lat
    return ee.Geometry.Rectangle(b, proj=None, geodesic=False)

def cloudmask_s2(img):
    # Use Sentinel-2 Scene Classification Layer (SCL) to mask clouds & shadows
    scl = img.select('SCL')
    mask = (scl.neq(3)  # cloud shadow
            .And(scl.neq(7))  # unclassified
            .And(scl.neq(8))  # medium prob. clouds
            .And(scl.neq(9))  # high prob. clouds
            .And(scl.neq(10)) # thin cirrus
           )
    return img.updateMask(mask)

def s2_composite(region, start, end, cloud_pct):
    col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
           .filterDate(start, end)
           .filterBounds(region)
           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_pct))
           .map(cloudmask_s2))
    # Select the 4 bands used downstream
    return col.median().select(['B2','B4','B8','B11'])

def slope_image(region):
    # Copernicus DEM 30m → slope
    dem = ee.Image('COPERNICUS/DEM/GLO-30').clip(region)
    slope = ee.Terrain.slope(dem).rename('slope')
    return slope

def ghi_image(region, ghi_start='2018-01-01', ghi_end='2024-12-31'):
    # ERA5-Land daily aggregation: surface_solar_radiation_downwards_sum (J/m^2 per day)
    era = (ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
           .filterDate(ghi_start, ghi_end)
           .filterBounds(region)
           .select('surface_solar_radiation_downwards_sum'))
    # Multi-year mean → proxy for long-term GHI
    return era.mean().rename('GHI')

def tile_grid(bbox, nx=8, ny=6):
    xmin, ymin, xmax, ymax = bbox
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    tiles = []
    t = 1
    # south→north rows (like your logs), west→east
    for j in range(ny):
        y0 = ymin + j*dy
        y1 = y0 + dy
        for i in range(nx):
            x0 = xmin + i*dx
            x1 = x0 + dx
            tiles.append((t, [x0, y0, x1, y1]))
            t += 1
    return tiles

def export_one(img, region, out_path, scale, crs):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # In some geemap versions, the signature is ee_export_image(ee_image, filename, ...)
    geemap.ee_export_image(
        img,                     # pass as positional, not image=...
        str(out_path),
        scale=scale,
        region=region,
        file_per_band=False,
        crs=crs
    )
    print(f"Data downloaded to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Download S2 composite, slope, and GHI (tiled) to data/raw/")
    p.add_argument("--export", choices=["local","assets"], default="local",
                   help="Use 'local' (download GeoTIFFs). (Assets mode not implemented here.)")
    p.add_argument("--small_aoi", action="store_true", help="Quick test: shrink AOI to a small box.")
    p.add_argument("--scale", type=int, default=EXPORT_SCALE, help="Export scale in meters (default from config).")
    p.add_argument("--crs", type=str, default="EPSG:3857", help="Output CRS (default: EPSG:3857).")
    p.add_argument("--nx", type=int, default=8, help="Tiles in X (default 8).")
    p.add_argument("--ny", type=int, default=6, help="Tiles in Y (default 6).")
    p.add_argument("--only-tiles", type=int, nargs="+", default=None,
                   help="Only download these tile indices (1-based), e.g. --only-tiles 25 28")
    p.add_argument("--resume", action="store_true",
                   help="Skip tiles/files that already exist.")
    p.add_argument("--ghi-start", type=str, default="2018-01-01", help="GHI start date (for ERA5-Land mean).")
    p.add_argument("--ghi-end", type=str, default="2024-12-31", help="GHI end date (for ERA5-Land mean).")
    args = p.parse_args()

    if args.export != "local":
        print("[WARN] --export assets is not implemented in this tiling helper. Using local download.")
    init_ee()

    # AOI
    bbox = list(AOI)
    if args.small_aoi:
        # shrink AOI around its center (~0.5° box)
        xmin, ymin, xmax, ymax = bbox
        cx, cy = (xmin+xmax)/2, (ymin+ymax)/2
        half = 0.25
        bbox = [cx-half, cy-half, cx+half, cy+half]
        print(f"[INFO] Folosesc AOI mic: {bbox}  (small=True)")
    else:
        print(f"[INFO] Folosesc AOI: {bbox}  (small=False)")

    # Build list of tiles
    tiles = tile_grid(bbox, nx=args.nx, ny=args.ny)
    if args.only_tiles:
        only = set(args.only_tiles)
        tiles = [t for t in tiles if t[0] in only]
        if not tiles:
            print("[ERROR] No tiles match --only-tiles selection.")
            sys.exit(2)

    # Pre-build static (time-invariant) sources where possible
    # Note: we still compute per-tile (clip) to reduce download size.
    failures = []
    total = len(tiles)
    for idx, tb in tiles:
        print(f"[INFO] Tile {idx:02d}/{total}: {tb}")
        region = rect_from_bbox(tb)

        # Compose images for this region
        try:
            s2 = s2_composite(region, START_DATE, END_DATE, CLOUD_PCT).clip(region)
            sl = slope_image(region)
            gi = ghi_image(region, args.ghi_start, args.ghi_end)

            # Paths
            s2_path = RAW_DIR / f"sentinel_composite_t{idx:02d}.tif"
            sl_path = RAW_DIR / f"slope_t{idx:02d}.tif"
            gi_path = RAW_DIR / f"ghi_t{idx:02d}.tif"

            # Resume: skip existing
            if args.resume and s2_path.exists() and sl_path.exists() and gi_path.exists():
                print(f"[SKIP] Tile {idx:02d} already complete (resume).")
                continue

            # Export each (skip individually if resume and file exists)
            if not (args.resume and s2_path.exists()):
                print(f"[INFO] Export LOCAL → {s2_path} (scale={args.scale} m, {args.crs})")
                export_one(s2, region, s2_path, args.scale, args.crs)
            else:
                print(f"[SKIP] {s2_path.name}")

            if not (args.resume and sl_path.exists()):
                print(f"[INFO] Export LOCAL → {sl_path} (scale={args.scale} m, {args.crs})")
                export_one(sl, region, sl_path, args.scale, args.crs)
            else:
                print(f"[SKIP] {sl_path.name}")

            if not (args.resume and gi_path.exists()):
                print(f"[INFO] Export LOCAL → {gi_path} (scale={args.scale} m, {args.crs})")
                export_one(gi, region, gi_path, args.scale, args.crs)
            else:
                print(f"[SKIP] {gi_path.name}")

        except Exception as e:
            print("[ERROR] Tile {:02d} failed: {}".format(idx, e))
            traceback.print_exc(limit=1)
            failures.append(idx)

    print("✔ Export LOCAL finalizat in data/raw/")
    if failures:
        print("[WARN] Failed tiles:", ", ".join(f"{i:02d}" for i in failures))
        print("      Re-run this command with:  --only-tiles", " ".join(str(i) for i in failures), "--resume")

if __name__ == "__main__":
    main()
