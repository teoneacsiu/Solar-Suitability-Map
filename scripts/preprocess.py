# scripts/preprocess.py
import argparse
import glob
import re
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window

from config import RAW_DIR as _RAW_DIR, PROC_DIR as _PROC_DIR

RAW_DIR = Path(_RAW_DIR)
PROC_DIR = Path(_PROC_DIR)
TMP_DIR = PROC_DIR / "_upright_tiles"
EPS = 1e-9

def _tile_index(path: str) -> int | None:
    m = re.search(r"_t(\d+)\.tif$", path)
    return int(m.group(1)) if m else None

def _list_tiles(kind: str) -> list[str]:
    return sorted(glob.glob(str(RAW_DIR / f"{kind}_t*.tif")))

def _report_missing(kind: str, expected=48) -> list[int]:
    files = _list_tiles(kind)
    got = {_tile_index(p) for p in files if _tile_index(p) is not None}
    missing = [i for i in range(1, expected + 1) if i not in got]
    if missing:
        print(f"[ERROR] {kind}: missing tiles: " + ", ".join(f"{i:02d}" for i in missing))
    else:
        print(f"[OK] {kind}: {len(got)}/{expected} tiles present")
    return missing

def _force_positive_e(t: Affine, height: int) -> Affine:
    e = float(t.e)
    if e < 0:
        e2 = -e
        if e2 < EPS:
            e2 = EPS
        return Affine(t.a, t.b, t.c, t.d, e2, t.f + e * (height - 1))
    elif abs(e) <= EPS:
        return Affine(t.a, t.b, t.c, t.d, EPS, t.f)
    else:
        return t

def _upright_copy(in_path: Path, out_path: Path, debug: bool=False):
    with rasterio.open(in_path) as src:
        t = src.transform
        e0 = float(t.e)
        data = src.read()
        meta = src.meta.copy()
        meta.update(compress="lzw", tiled=True, bigtiff="IF_SAFER")
        new_t = _force_positive_e(t, src.height)
        if e0 < 0:
            data = data[:, ::-1, :]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        meta.update(transform=new_t)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data)
    with rasterio.open(out_path) as ds:
        if not (float(ds.transform.e) > 0):
            raise SystemExit(f"[FATAL] e<=0 after upright copy: {out_path}")
        if debug:
            print(f"  - {in_path.name}: e_in={e0:.6f} -> e_out={float(ds.transform.e):.6f}")

def _prepare_upright(kind: str, debug: bool) -> list[Path]:
    srcs = _list_tiles(kind)
    if not srcs:
        raise SystemExit(f"[FATAL] No tiles match {kind}_t*.tif in {RAW_DIR}")
    out_dir = TMP_DIR / kind
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    print(f"[INFO] Normalizing {kind} tiles → {out_dir}")
    for p in srcs:
        i = _tile_index(p)
        q = out_dir / f"{kind}_t{i:02d}.tif"
        _upright_copy(Path(p), q, debug=debug)
        out.append(q)
    return sorted(out)

def _mosaic_manual(kind: str, out_path: Path, debug: bool):
    fixed = _prepare_upright(kind, debug=debug)
    print(f"[INFO] Mosaicking {kind} manually (no rasterio.merge)")

    # Use first tile as reference
    with rasterio.open(fixed[0]) as ref:
        a = float(ref.transform.a)
        e = float(ref.transform.e)  # strictly > 0 after normalization
        if not (a > 0 and e > 0):
            raise SystemExit("[FATAL] Reference tile has invalid pixel size.")
        crs = ref.crs
        count = ref.count
        dtype = ref.dtypes[0]
        nodata = ref.nodata

    # Compute origin (min c,f) and mosaic size in pixels
    ref_c = min(rasterio.open(fp).transform.c for fp in fixed)
    ref_f = min(rasterio.open(fp).transform.f for fp in fixed)

    max_col = 0
    max_row = 0
    offsets = []
    for fp in fixed:
        with rasterio.open(fp) as ds:
            t = ds.transform
            # integer offsets on the reference grid
            col_off = int(round((t.c - ref_c) / a))
            row_off = int(round((t.f - ref_f) / e))
            offsets.append((fp, col_off, row_off, ds.width, ds.height))
            max_col = max(max_col, col_off + ds.width)
            max_row = max(max_row, row_off + ds.height)

    transform_out = Affine(a, 0.0, ref_c, 0.0, e, ref_f)

    profile = {
        "driver": "GTiff",
        "height": max_row,
        "width": max_col,
        "count": count,
        "dtype": dtype,
        "crs": crs,
        "transform": transform_out,
        "compress": "lzw",
        "tiled": True,
        "bigtiff": "IF_SAFER",
        "nodata": nodata,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        # initialize with nodata (if numeric)
        if nodata is not None and np.issubdtype(np.dtype(dtype), np.number):
            for b in range(1, count + 1):
                dst.write(np.full((max_row, max_col), nodata, dtype=dtype), indexes=b)

        # paste each tile window-by-window
        for fp, col_off, row_off, w, h in offsets:
            window = Window(col_off, row_off, w, h)
            with rasterio.open(fp) as ds:
                data = ds.read()
                dst.write(data, window=window)

    print(f"[OK] {kind} → {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Preprocess: upright tiles and manual mosaic.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    PROC_DIR.mkdir(parents=True, exist_ok=True)

    miss_s2 = _report_missing("sentinel_composite")
    miss_slp = _report_missing("slope")
    miss_ghi = _report_missing("ghi")
    if miss_s2 or miss_slp or miss_ghi:
        print("\n[HINT] Re-download only missing tiles, e.g.:")
        if miss_s2:
            print("  python -m scripts.download_data --export local --only-tiles " + " ".join(map(str, miss_s2)) + " --resume")
        if miss_slp:
            print("  python -m scripts.download_data --export local --only-tiles " + " ".join(map(str, miss_slp)) + " --resume")
        if miss_ghi:
            print("  python -m scripts.download_data --export local --only-tiles " + " ".join(map(str, miss_ghi)) + " --resume")
        raise SystemExit("[FATAL] Missing tiles detected — fix first, then re-run.")

    _mosaic_manual("sentinel_composite", PROC_DIR / "sentinel_composite.tif", args.debug)
    _mosaic_manual("slope",              PROC_DIR / "slope.tif",              args.debug)
    _mosaic_manual("ghi",                PROC_DIR / "ghi.tif",                args.debug)

    print(f"[CLEANUP] Working copies kept at {TMP_DIR} (safe to delete later).")
    print("Done preprocessing.")

if __name__ == "__main__":
    main()
