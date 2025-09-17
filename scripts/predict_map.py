# scripts/predict_map.py
import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window
from joblib import load

from config import PROC_DIR, MODEL_PATH

PROC_DIR = Path(PROC_DIR)

def _pick_features(B2, B4, B8, B11, slope, ghi, need):
    """Return a (H,W,need) feature cube that matches the trained model."""
    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    if need == 4:
        # Common minimal set: NDVI + terrain + irradiance + SWIR
        return np.stack([ndvi, slope, ghi, B11], axis=-1)
    if need == 6:
        # Raw spectral + terrain + irradiance
        return np.stack([B2, B4, B8, B11, slope, ghi], axis=-1)
    if need == 7:
        # Raw spectral + terrain + irradiance + NDVI
        return np.stack([B2, B4, B8, B11, slope, ghi, ndvi], axis=-1)
    raise ValueError(
        f"Model expects {need} features, but this script supports only 4/6/7. "
        "If you used a different recipe, expose the same feature builder here."
    )

def predict_streamed(chunk, out_dtype):
    s2_path   = PROC_DIR / "sentinel_composite.tif"  # B2,B4,B8,B11
    slope_path= PROC_DIR / "slope.tif"
    ghi_path  = PROC_DIR / "ghi.tif"
    out_tif   = PROC_DIR / "suitability_map.tif"

    model = load(MODEL_PATH)
    n_feat = int(getattr(model, "n_features_in_", 0))
    if n_feat not in (4, 6, 7):
        raise SystemExit(
            f"[FATAL] Trained model expects {n_feat} features. "
            "Update _pick_features() to match your training recipe."
        )

    with rasterio.open(s2_path) as s2, \
         rasterio.open(slope_path) as rs, \
         rasterio.open(ghi_path) as rg:

        # Basic sanity
        if not (s2.width == rs.width == rg.width and s2.height == rs.height == rg.height):
            raise SystemExit("[FATAL] Raster shapes differ; rerun preprocessing.")
        if not (s2.crs == rs.crs == rg.crs):
            raise SystemExit("[FATAL] Raster CRS differ; rerun preprocessing.")

        profile = s2.profile.copy()
        profile.update(
            count=1,
            dtype=out_dtype,
            compress="lzw",
            tiled=True,
            bigtiff="IF_SAFER",
            nodata=0
        )

        H, W = s2.height, s2.width
        with rasterio.open(out_tif, "w", **profile) as dst:
            # iterate over windows
            for row0 in range(0, H, chunk):
                h = min(chunk, H - row0)
                for col0 in range(0, W, chunk):
                    w = min(chunk, W - col0)
                    win = Window(col_off=col0, row_off=row0, width=w, height=h)

                    # read windowed data as float32
                    B2  = s2.read(1, window=win, out_dtype="float32")
                    B4  = s2.read(2, window=win, out_dtype="float32")
                    B8  = s2.read(3, window=win, out_dtype="float32")
                    B11 = s2.read(4, window=win, out_dtype="float32")
                    slp = rs.read(1, window=win, out_dtype="float32")
                    ghi = rg.read(1, window=win, out_dtype="float32")

                    # build features matching the trained model
                    feats = _pick_features(B2, B4, B8, B11, slp, ghi, n_feat)

                    # validity mask (avoid NaNs/infs)
                    valid = np.isfinite(feats).all(axis=-1)
                    flat = feats.reshape(-1, n_feat)

                    proba = np.zeros(flat.shape[0], dtype="float32")
                    if valid.any():
                        valid_idx = valid.reshape(-1)
                        pred = model.predict_proba(flat[valid_idx])[:, 1].astype("float32")
                        proba[valid_idx] = pred
                    proba = proba.reshape(h, w)

                    # cast/scale for output
                    if out_dtype == "float32":
                        out_block = proba
                    else:
                        # uint8 heatmap 0..255
                        out_block = (np.clip(proba, 0, 1) * 255.0).astype("uint8")

                    dst.write(out_block, 1, window=win)

    print(f"✔ Saved: {out_tif}")

def main():
    ap = argparse.ArgumentParser(description="Memory-safe prediction over the full mosaic.")
    ap.add_argument("--export", choices=["local","assets"], default="local",
                    help="(assets not used here; kept for CLI parity)")
    ap.add_argument("--chunk", type=int, default=512,
                    help="Tile window size in pixels (e.g., 512 or 1024). Smaller = lower RAM.")
    ap.add_argument("--out-dtype", choices=["float32","uint8"], default="float32",
                    help="Output dtype. uint8 writes a compact heatmap 0..255.")
    args = ap.parse_args()

    predict_streamed(chunk=args.chunk, out_dtype=args.out_dtype)

if __name__ == "__main__":
    main()
