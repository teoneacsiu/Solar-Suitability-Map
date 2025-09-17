# scripts/shap_explain.py
# --------------------------------------------------------------
# SHAP pentru modelul RF pe un eșantion de pixeli.
# Folosește aceleași feature-uri ca predicția (ex.: NDVI/NDBI/NDWI, slope, GHI).
# Salvează summary plot + bar importances (PNG) în data/metrics/.
# Dacă 'shap' nu e instalat, revine la permutation_importance.
# --------------------------------------------------------------

import argparse
from pathlib import Path
import numpy as np
import joblib
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt

def reproject_to_ref(src_arr, src_transform, src_crs, ref_transform, ref_crs, ref_shape):
    dst = np.full(ref_shape, np.nan, dtype=np.float32)
    reproject(
        source=src_arr.astype(np.float32),
        destination=dst,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=ref_transform, dst_crs=ref_crs,
        resampling=Resampling.bilinear,
    )
    return dst

def main():
    p = argparse.ArgumentParser(description="SHAP explanations for RF model.")
    p.add_argument("--model", default="models/rf_model.joblib")
    p.add_argument("--map", required=True, help="Reference raster (to get grid)")
    p.add_argument("--s2",  required=True)
    p.add_argument("--slope", required=True)
    p.add_argument("--ghi", required=False, help="Optional GHI raster if used as feature")
    p.add_argument("--sample", type=int, default=5000)
    p.add_argument("--outdir", default="data/metrics")
    args = p.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    clf = joblib.load(args.model)

    # FEATURES din S2 + slope (+ optional GHI)
    with rasterio.open(args.map) as ref: H,W = ref.height, ref.width; ref_tr, ref_crs = ref.transform, ref.crs

    with rasterio.open(args.s2) as s2src:
        B2,B4,B8,B11 = s2src.read().astype("float32")
        s2_tr, s2_crs = s2src.transform, s2src.crs

    ndvi = (B8 - B4) / (B8 + B4 + 1e-6)
    ndbi = (B11 - B8) / (B11 + B8 + 1e-6)
    ndwi = (B2 - B11) / (B2 + B11 + 1e-6)

    ndvi = reproject_to_ref(ndvi, s2_tr, s2_crs, ref_tr, ref_crs, (H,W))
    ndbi = reproject_to_ref(ndbi, s2_tr, s2_crs, ref_tr, ref_crs, (H,W))
    ndwi = reproject_to_ref(ndwi, s2_tr, s2_crs, ref_tr, ref_crs, (H,W))

    with rasterio.open(args.slope) as ss: slope = reproject_to_ref(ss.read(1).astype("float32"), ss.transform, ss.crs, ref_tr, ref_crs, (H,W))

    feats = [ndvi, ndbi, ndwi, slope]
    names = ["NDVI","NDBI","NDWI","SLOPE"]

    if args.ghi:
        with rasterio.open(args.ghi) as gs:
            ghi = reproject_to_ref(gs.read(1).astype("float32"), gs.transform, gs.crs, ref_tr, ref_crs, (H,W))
        feats.append(ghi); names.append("GHI")

    X = np.stack(feats, axis=-1)
    valid = np.all(np.isfinite(X), axis=-1)
    rows, cols = np.where(valid)
    rng = np.random.default_rng(0)
    take = rng.choice(len(rows), size=min(args.sample, len(rows)), replace=False)
    Xs = X[rows[take], cols[take], :]

    # SHAP (dacă există)
    try:
        import shap
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xs)
        shap.summary_plot(sv, Xs, feature_names=names, show=False)
        plt.tight_layout(); plt.savefig(outdir/"shap_summary.png", dpi=150); plt.close()

        shap.summary_plot(sv, Xs, feature_names=names, plot_type="bar", show=False)
        plt.tight_layout(); plt.savefig(outdir/"shap_bar.png", dpi=150); plt.close()
        print("✔ SHAP plots saved.")
    except Exception as e:
        print(f"[WARN] SHAP unavailable ({e}); using permutation importance.")
        try:
            from sklearn.inspection import permutation_importance
            r = permutation_importance(clf, Xs, np.ones(len(Xs)), n_repeats=5, random_state=0)
            imp = r.importances_mean
            order = np.argsort(imp)[::-1]
            plt.figure(figsize=(4,3), dpi=150)
            plt.bar([names[i] for i in order], imp[order])
            plt.ylabel("Permutation importance"); plt.tight_layout()
            plt.savefig(outdir/"perm_importance.png"); plt.close()
            print("✔ Permutation importance saved.")
        except Exception as e2:
            print(f"[WARN] Permutation importance failed: {e2}")

if __name__ == "__main__":
    main()
