# scripts/postprocess_zones.py
# --------------------------------------------------------------
# Make "suitability zones" from a probability/score GeoTIFF:
#  - optional percentile normalization (qmin..qmax)
#  - threshold -> binary mask
#  - morphology: open/close (px), fill holes
#  - remove small regions by MIN AREA (km²) robust to CRS
#  - write: normalized (optional), final mask (uint8), GeoJSON polygons
# GeoJSON is in EPSG:4326 so Folium/Leaflet can load it directly.
# --------------------------------------------------------------

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds, transform_geom
from rasterio.transform import array_bounds
from rasterio import features as rio_features


# ---------------- utils ----------------

def disk_kernel(radius: int) -> np.ndarray:
    """Binary disk (radius in pixels) for morphology."""
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y) <= (radius * radius)


def pixel_area_km2(transform, crs, shape_hw):
    """
    Area per pixel in km², robust even if CRS is geographic (degrees).
    Uses bounds densified to EPSG:3857 to derive px size in meters.
    """
    try:
        is_projected = getattr(crs, "is_projected", False)
    except Exception:
        is_projected = False

    if is_projected or "3857" in str(crs) or "3395" in str(crs):
        px_w = abs(transform.a)
        px_h = abs(transform.e)
        return (px_w * px_h) / 1e6

    H, W = shape_hw
    b = array_bounds(H, W, transform)  # in source CRS
    x0, y0, x1, y1 = transform_bounds(crs, "EPSG:3857", *b, densify_pts=21)
    px_w_m = abs((x1 - x0) / W)
    px_h_m = abs((y1 - y0) / H)
    return (px_w_m * px_h_m) / 1e6


def write_tif_like(profile_or_dataset, out_path: Path, arr: np.ndarray,
                   dtype="float32", nodata=None):
    """
    Write a GeoTIFF using the same spatial profile. Accepts either:
      - a dict/profile-like mapping
      - an open rasterio dataset (has .profile)
    Robust to rasterio.profiles.Profile instances.
    """
    base_prof = None
    # Try mapping-like first
    try:
        # dict(Profile) works and yields a plain mapping
        base_prof = dict(profile_or_dataset)
    except Exception:
        # Not a mapping -> maybe an open dataset
        try:
            base_prof = profile_or_dataset.profile.copy()
        except Exception:
            raise TypeError("Unsupported profile_or_dataset type for write_tif_like")

    prof = dict(base_prof)
    prof.update({
        "driver": "GTiff",
        "count": 1,
        "dtype": dtype,
        "compress": "lzw",
    })
    if nodata is not None:
        prof["nodata"] = nodata
    else:
        if str(dtype).startswith("float"):
            prof["nodata"] = np.nan

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(out_path), "w", **prof) as dst:
        dst.write(arr.astype(dtype), 1)


# ---------------- main ----------------

def main():
    p = argparse.ArgumentParser(
        description="Post-process a suitability GeoTIFF into contiguous zones."
    )
    p.add_argument("--in", dest="in_path", required=True,
                   help="Input suitability GeoTIFF (probability/score).")
    p.add_argument("--normalize", action="store_true",
                   help="Normalize by percentiles qmin..qmax before threshold.")
    p.add_argument("--qmin", type=float, default=2.0,
                   help="Lower percentile for normalization (default 2).")
    p.add_argument("--qmax", type=float, default=98.0,
                   help="Upper percentile for normalization (default 98).")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Threshold on (normalized) values to keep (default 0.5).")
    p.add_argument("--open", dest="open_rad", type=int, default=0,
                   help="Binary opening radius in pixels (e.g., 1).")
    p.add_argument("--close", dest="close_rad", type=int, default=0,
                   help="Binary closing radius in pixels (e.g., 2).")
    p.add_argument("--fill-holes", action="store_true",
                   help="Fill internal holes in kept regions.")
    p.add_argument("--min-area-km2", type=float, default=0.0,
                   help="Remove regions smaller than this area (km²).")
    p.add_argument("--out-mask", required=True,
                   help="Output GeoTIFF mask (uint8, 0/1).")
    p.add_argument("--out-geojson", required=True,
                   help="Output GeoJSON of zones (EPSG:4326).")
    p.add_argument("--write-normalized", action="store_true",
                   help="Also write a normalized float GeoTIFF next to input.")
    p.add_argument("--connectivity", type=int, choices=[4, 8], default=8,
                   help="Connectivity for region labeling (default 8).")

    args = p.parse_args()

    in_path = Path(args.in_path)
    out_mask_path = Path(args.out_mask)
    out_geojson_path = Path(args.out_geojson)

    # --- read input
    with rasterio.open(str(in_path)) as src:
        arr = src.read(1).astype("float32")
        transform = src.transform
        crs = src.crs
        profile_ref = dict(src.profile)  # force plain dict to avoid Profile object issues

    valid = np.isfinite(arr)
    if not np.any(valid):
        raise SystemExit("[FATAL] All pixels are NaN in the input raster.")

    # --- optional normalization
    arr_work = arr.copy()
    if args.normalize:
        valid_vals = arr_work[valid]
        vmin = np.percentile(valid_vals, args.qmin)
        vmax = np.percentile(valid_vals, args.qmax)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = np.nanmin(valid_vals), np.nanmax(valid_vals)
        arr_norm = (arr_work - vmin) / (vmax - vmin + 1e-9)
        arr_norm = np.clip(arr_norm, 0.0, 1.0)
        arr_work[~valid] = np.nan
        if args.write_normalized:
            norm_path = in_path.with_name(in_path.stem + "_normalized.tif")
            print(f"[INFO] Writing normalized raster → {norm_path}")
            write_tif_like(profile_ref, norm_path,
                           np.where(valid, arr_norm, np.nan),
                           dtype="float32", nodata=np.nan)
        use = arr_norm
    else:
        use = arr_work

    # --- threshold to binary
    mask = (use >= float(args.threshold)) & valid

    # --- morphology
    try:
        from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes, label
        have_scipy = True
    except Exception as e:
        print(f"[WARN] SciPy not available ({e}); skipping morphology + labeling.")
        have_scipy = False
        label = None  # keep linters calm

    if have_scipy:
        if args.open_rad > 0:
            mask = binary_opening(mask, structure=disk_kernel(args.open_rad))
        if args.close_rad > 0:
            mask = binary_closing(mask, structure=disk_kernel(args.close_rad))
        if args.fill_holes:
            mask = binary_fill_holes(mask)

    # --- remove small regions by area (km²)
    px_km2 = pixel_area_km2(transform, crs, mask.shape)
    if args.min_area_km2 > 0 and have_scipy:
        conn = np.array([[0, 1, 0],
                         [1, 1, 1],
                         [0, 1, 0]], dtype=bool) if args.connectivity == 4 else np.ones((3, 3), dtype=bool)
        labels, nlab = label(mask, structure=conn)
        if nlab > 0:
            counts = np.bincount(labels.ravel())
            min_pixels = int(np.ceil(args.min_area_km2 / max(px_km2, 1e-12)))
            keep_ids = [i for i in range(1, nlab + 1) if counts[i] >= min_pixels]
            if len(keep_ids) == 0:
                print("[INFO] After min-area filter, no regions remain; mask becomes empty.")
                mask[:] = False
            else:
                keep_mask = np.isin(labels, keep_ids)
                mask = keep_mask
                labels = labels * keep_mask  # zero out removed
        else:
            print("[INFO] No labeled regions found before area filtering.")
            labels = np.zeros_like(mask, dtype=np.int32)
    else:
        if have_scipy:
            conn = np.array([[0, 1, 0],
                             [1, 1, 1],
                             [0, 1, 0]], dtype=bool) if args.connectivity == 4 else np.ones((3, 3), dtype=bool)
            labels, nlab = label(mask, structure=conn)
            labels = labels * (labels > 0)
        else:
            labels = (mask.astype(np.uint8))  # fallback

    # --- write mask GeoTIFF (uint8)
    print(f"[INFO] Writing mask → {out_mask_path}")
    write_tif_like(profile_ref, out_mask_path, mask.astype("uint8"), dtype="uint8", nodata=0)

    # --- vectorize to GeoJSON (EPSG:4326), one feature per labeled region
    features = []
    if labels.dtype.kind in "iu" and getattr(labels, "max", lambda: 0)() > 1:
        vals = np.unique(labels)
        vals = vals[vals > 0]
        counts = np.bincount(labels.ravel())
        for geom, val in rio_features.shapes(labels, transform=transform):
            if val == 0:
                continue
            try:
                geom84 = transform_geom(crs, "EPSG:4326", geom, precision=6)
            except Exception:
                geom84 = geom  # fallback
            area_km2 = float(counts[int(val)] * px_km2)
            features.append({
                "type": "Feature",
                "properties": {"id": int(val), "area_km2": area_km2},
                "geometry": geom84
            })
    else:
        for geom, val in rio_features.shapes(mask.astype(np.uint8), transform=transform):
            if val != 1:
                continue
            try:
                geom84 = transform_geom(crs, "EPSG:4326", geom, precision=6)
            except Exception:
                geom84 = geom
            features.append({
                "type": "Feature",
                "properties": {"area_km2": None},
                "geometry": geom84
            })

    fc = {"type": "FeatureCollection", "features": features}
    out_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    out_geojson_path.write_text(json.dumps(fc))

    # --- summary
    kept_px = int(np.count_nonzero(mask))
    total_px = int(np.count_nonzero(np.isfinite(arr)))
    kept_km2 = kept_px * px_km2
    total_km2 = total_px * px_km2
    print("[INFO] Summary")
    print(f"  Pixels kept: {kept_px} / {total_px}  ({100.0*kept_px/max(total_px,1):.2f}%)")
    print(f"  Area kept:   {kept_km2:,.1f} km² out of {total_km2:,.1f} km²".replace(",", " "))


if __name__ == "__main__":
    main()
