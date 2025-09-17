# run_all.py
# --------------------------------------------------------------------------------------
# Full pipeline runner for the Solar Suitability project:
# download_data -> preprocess -> extract_features -> generate_labels -> train_model
# -> predict_map -> visualize (HTML + overlays + metrics card).
#
# Notes:
# - Requires Earth Engine auth + CLI in PATH (even when exporting locally).
# - Detects and passes admin overlays (counties/cities) and metrics if present.
# - Matches visualize.py arguments: --viz-mode -> ["prob","binary","auto"] is mapped
#   to visualize's ["prob","classes"] (binary -> classes; auto -> prob).
# - By default, enables constraints overlay in the HTML (can be disabled via flag).
# --------------------------------------------------------------------------------------

import argparse
import sys
import time
import subprocess
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = ROOT / "scripts"
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
ADMIN_DIR = ROOT / "data" / "admin"
METRICS_DIR = ROOT / "data" / "metrics"

def sh(cmd: list[str], cwd: Path = ROOT) -> None:
    """Run a command in a separate process; stop on error with a clear message."""
    print("\n" + "=" * 88)
    print(">>>", " ".join(cmd))
    print("=" * 88)
    res = subprocess.run(cmd, cwd=str(cwd))
    if res.returncode != 0:
        print(f"\n[ERROR] Exit code {res.returncode}: {' '.join(cmd)}")
        sys.exit(res.returncode)

def ensure_dirs_and_pkg() -> None:
    """Create needed folders and make scripts/ a package so `-m` works."""
    (ROOT / "models").mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ADMIN_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    init_file = SCRIPTS_DIR / "__init__.py"
    if not init_file.exists():
        init_file.write_text("")

def preflight(require_ee: bool = True) -> None:
    """Basic checks: interpreter, core Python modules, config, (optionally) earthengine CLI."""
    print(f"[INFO] Python: {sys.executable}")

    required = ["numpy", "rasterio", "sklearn", "joblib", "PIL", "folium"]
    optional = ["geopandas", "matplotlib", "scipy"]  # used for vector overlays, metrics, buffers

    missing = []
    for m in required:
        try:
            __import__(m if m != "PIL" else "PIL.Image")
        except Exception:
            missing.append(m)
    if missing:
        print(f"[ERROR] Missing required modules: {', '.join(missing)}")
        print("        Install with:  python -m pip install -r requirements.txt")
        sys.exit(1)

    missing_opt = []
    for m in optional:
        try:
            __import__(m)
        except Exception:
            missing_opt.append(m)
    if missing_opt:
        print(f"[WARN] Optional modules not found (some features may be skipped): {', '.join(missing_opt)}")

    # config + EE project
    try:
        import config  # type: ignore
        ee_project = getattr(config, "EE_PROJECT", None)
        if not ee_project or not isinstance(ee_project, str) or not ee_project.strip():
            print("[ERROR] EE_PROJECT is not set in config.py.")
            print('        Example: EE_PROJECT = "your-gcp-project-id"')
            sys.exit(1)
        print(f"[INFO] EE_PROJECT: {ee_project}")
    except Exception as e:
        print(f"[ERROR] Cannot import config.py: {e}")
        sys.exit(1)

    if require_ee:
        try:
            subprocess.run(["earthengine", "help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception:
            print("[ERROR] `earthengine` CLI not found in PATH.")
            sys.exit(1)

def safe_unlink(path: Path, retries: int = 6, delay: float = 0.5) -> None:
    """Delete a file if exists, with a few retries (helps on Windows/WSL locks)."""
    import time as _t
    if not path.exists():
        return
    for _ in range(retries):
        try:
            path.unlink()
            return
        except PermissionError:
            _t.sleep(delay)
    try:
        path.rename(path.with_suffix(path.suffix + f".old.{int(time.time())}"))
    except Exception:
        pass

def pick_visualize_map() -> Path:
    """Prefer processed/suitability_adjusted.tif if exists; else processed/suitability_map.tif."""
    adjusted = PROC_DIR / "suitability_adjusted.tif"
    fallback = PROC_DIR / "suitability_map.tif"
    if adjusted.exists():
        return adjusted
    if fallback.exists():
        return fallback
    # Last resort: raw model output name used earlier (if any)
    other = PROC_DIR / "predict.tif"
    return other

def main():
    parser = argparse.ArgumentParser(
        description="Run full Solar Suitability pipeline (download → preprocess → features → labels → train → predict → visualize)."
    )
    # data export mode (download/predict)
    parser.add_argument("--mode", choices=["local", "assets"], default="local",
                        help="Export local GeoTIFFs (local) or upload to Earth Engine Assets (assets).")
    # labels generation
    parser.add_argument("--n_pos", type=int, default=300, help="Number of positive labels to sample.")
    parser.add_argument("--n_neg", type=int, default=300, help="Number of negative labels to sample.")
    parser.add_argument("--ghi_q", type=float, default=0.70, help="GHI quantile threshold for 'high'.")
    parser.add_argument("--reuse-labels", action="store_true", help="Reuse existing labels.csv if present.")
    # quick sanity AOI
    parser.add_argument("--quick-test", action="store_true",
                        help="Use a small AOI (~20x20 km) for a fast sanity run (download).")
    # steps toggles
    parser.add_argument("--skip-download", action="store_true", help="Skip download_data step.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocess step.")
    parser.add_argument("--skip-extract", action="store_true", help="Skip extract_features step.")
    parser.add_argument("--skip-train", action="store_true", help="Skip train_model step.")
    parser.add_argument("--skip-predict", action="store_true", help="Skip predict_map step.")
    parser.add_argument("--skip-visualize", action="store_true", help="Skip visualize step.")
    # visualization options (mapped to scripts.visualize)
    parser.add_argument("--viz-mode", choices=["auto", "prob", "binary"], default="auto",
                        help="Visualization mode: auto→prob; prob=gradient; binary=classes.")
    parser.add_argument("--viz-threshold", type=float, default=0.5,
                        help="Threshold for binary/classes view.")
    parser.add_argument("--viz-basemap", default="CartoDB.Positron",
                        help="Basemap for HTML (e.g., CartoDB.Positron, OpenStreetMap, Esri.WorldImagery or custom tiles URL).")
    parser.add_argument("--viz-constraints", action="store_true",
                        help="Also overlay condition layers (water/urban/forest/slope).")
    parser.add_argument("--viz-no-admin", action="store_true",
                        help="Do not add counties/cities overlays even if files are present.")
    parser.add_argument("--viz-out-suffix", default="",
                        help="Suffix for output filenames (e.g. 'explain').")
    # pass-through thresholds for constraints overlay
    parser.add_argument("--slope-max",   type=float, default=7.0)
    parser.add_argument("--ndvi-forest", type=float, default=0.55)
    parser.add_argument("--ndwi-water",  type=float, default=0.05)
    parser.add_argument("--ndbi-urban",  type=float, default=0.05)
    parser.add_argument("--buffer-water-px", type=int, default=1)
    parser.add_argument("--buffer-urban-px", type=int, default=1)
    # metrics card
    parser.add_argument("--metrics", default=str(METRICS_DIR / "metrics.json"),
                        help="Path to metrics.json to display an evaluation card in HTML (if file exists).")

    args = parser.parse_args()

    t0 = time.time()
    ensure_dirs_and_pkg()
    preflight(require_ee=True)  # EE required for download/predict

    # 1) Download
    if not args.skip_download:
        dl_cmd = [sys.executable, "-m", "scripts.download_data", "--export", args.mode]
        if args.quick_test:
            dl_cmd.append("--small_aoi")
        sh(dl_cmd)
    else:
        print("[INFO] Skipping download_data (per flag).")

    # 2) Preprocess
    if not args.skip_preprocess:
        sh([sys.executable, "-m", "scripts.preprocess"])
    else:
        print("[INFO] Skipping preprocess (per flag).")

    # 3) Feature extraction
    if not args.skip_extract:
        sh([sys.executable, "-m", "scripts.extract_features"])
    else:
        print("[INFO] Skipping extract_features (per flag).")

    # 4) Generate labels (auto) unless reusing
    labels_path = PROC_DIR / "labels.csv"
    if args.reuse_labels and labels_path.exists():
        print(f"[INFO] Reusing existing labels: {labels_path}")
    else:
        sh([
            sys.executable, "-m", "scripts.generate_labels",
            "--n_pos", str(args.n_pos),
            "--n_neg", str(args.n_neg),
            "--ghi_q", str(args.ghi_q)
        ])
        if not labels_path.exists():
            print("[ERROR] generate_labels.py ran but data/processed/labels.csv is missing.")
            sys.exit(1)

    # 5) Train model
    if not args.skip_train:
        sh([sys.executable, "-m", "scripts.train_model"])
    else:
        print("[INFO] Skipping train_model (per flag).")

    # 6) Predict map (local or assets)
    if not args.skip_predict:
        sh([sys.executable, "-m", "scripts.predict_map", "--export", args.mode])
    else:
        print("[INFO] Skipping predict_map (per flag).")

    # 7) Visualization (always regenerates PNG/HTML unless skipped)
    if not args.skip_visualize:
        # Clean old viz artifacts so they are definitely updated
        safe_unlink(PROC_DIR / "suitability_colored.png")
        safe_unlink(PROC_DIR / "suitability_map.html")

        # Build visualize command
        viz_cmd = [sys.executable, "-m", "scripts.visualize"]

        # Map file to show
        map_tif = pick_visualize_map()
        if map_tif.exists():
            viz_cmd += ["--map", str(map_tif)]
        else:
            # Let visualize fall back / error with a clear message
            pass

        # viz mode mapping
        vm = args.viz_mode
        if vm == "binary":
            viz_cmd += ["--viz-mode", "classes"]
        elif vm in ("prob", "auto"):
            # 'auto' -> use 'prob' as default presentation
            viz_cmd += ["--viz-mode", "prob"]
        else:
            viz_cmd += ["--viz-mode", vm]

        viz_cmd += [
            "--threshold", str(args.viz_threshold),
            "--basemap", args.viz_basemap,
        ]

        # Constraints overlay + thresholds
        if args.viz_constraints:
            viz_cmd.append("--constraints")
            viz_cmd += [
                "--slope-max", str(args.slope_max),
                "--ndvi-forest", str(args.ndvi_forest),
                "--ndwi-water", str(args.ndwi_water),
                "--ndbi-urban", str(args.ndbi_urban),
                "--buffer-water-px", str(args.buffer_water_px),
                "--buffer-urban-px", str(args.buffer_urban_px),
            ]

        # Counties / Cities overlays (if files exist and not disabled)
        if not args.viz_no_admin:
            counties = ADMIN_DIR / "ro_counties.geojson"
            cities   = ADMIN_DIR / "ro_cities.geojson"
            if counties.exists():
                viz_cmd += ["--counties", str(counties)]
            else:
                print(f"[INFO] Counties GeoJSON not found at {counties} (overlay skipped).")
            if cities.exists():
                viz_cmd += ["--cities", str(cities)]
            else:
                print(f"[INFO] Cities GeoJSON not found at {cities} (overlay skipped).")
        else:
            print("[INFO] Admin overlays disabled via --viz-no-admin.")

        # Metrics card (only if file exists)
        metrics_path = Path(args.metrics) if args.metrics else None
        if metrics_path and metrics_path.exists():
            viz_cmd += ["--metrics", str(metrics_path)]
        else:
            if metrics_path:
                print(f"[INFO] Metrics file not found at {metrics_path} (metrics card skipped).")

        # Output suffix for HTML/PNG filenames
        if args.viz_out_suffix:
            viz_cmd += ["--out-suffix", args.viz_out_suffix]

        sh(viz_cmd)

    dt = time.time() - t0
    print("\n" + "=" * 88)
    print(f"🎉 DONE! Full pipeline finished in {dt:.1f} s")
    print("Outputs:")
    print(f" - Labels:   {labels_path}")
    print(f" - Model:    {ROOT / 'models' / 'rf_model.joblib'}")
    print(f" - Map TIF:  {PROC_DIR / 'suitability_map.tif'}")
    if not args.skip_visualize:
        suffix = f"_{args.viz_out_suffix}" if args.viz_out_suffix else ""
        print(f" - PNG:      {PROC_DIR / ('suitability_colored' + suffix + '.png')}")
        print(f" - HTML:     {PROC_DIR / ('suitability_map' + suffix + '.html')}")
    if args.mode == "assets":
        print(" - Upload to EE Assets started as a Task. Check Code Editor → Tasks.")
    print("=" * 88)

if __name__ == "__main__":
    main()
