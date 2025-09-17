# Solar Suitability (ML/DL – Energy)

This project generates a **solar farm suitability map** using open satellite data (Sentinel-2, DEM) and a **Random Forest** model. The goal is to produce a GeoTIFF with per-pixel probabilities (0–1) indicating how suitable each location is for PV development.

---

## Quickstart (copy–paste)

```sh
# 1) Create & activate a virtual environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) Configure AOI & parameters
# Edit config.py -> AOI, START_DATE, END_DATE, CLOUD_PCT, EXPORT_SCALE

# 4) (Optional) Authenticate Google Earth Engine if download_data.py uses GEE
pip install earthengine-api geemap
earthengine authenticate --auth_mode=oauth
# If needed:
# gcloud auth application-default login
# earthengine set_project <YOUR_EE_PROJECT_ID>

# 5) Run the pipeline step-by-step
python scripts/download_data.py
python scripts/preprocess.py
python scripts/extract_features.py
# Prepare labels (see 'Create labels' below), then:
python scripts/train_model.py
python scripts/predict_map.py
python scripts/visualize.py
```

> One-shot runner (executes all steps in order):
```sh
python - <<'PY'
import subprocess
for step in [
  "scripts/download_data.py",
  "scripts/preprocess.py",
  "scripts/extract_features.py",
  "scripts/train_model.py",
  "scripts/predict_map.py",
  "scripts/visualize.py",
]:
    print(f"==> Running {step}")
    subprocess.check_call(["python", step])
print("All steps completed.")
PY
```

---

## Description

The pipeline takes an Area of Interest (AOI), builds a cloud-filtered Sentinel-2 composite, derives **NDVI**, **NDBI**, **slope** from a DEM, and a **GHI proxy**. Features are stacked, a Random Forest is trained on labeled samples, and predictions are exported as a georeferenced raster.

Main stages:
1. **Download** raw rasters (Sentinel-2 composite, DEM→slope, GHI proxy).
2. **Preprocess** (align/resample to a common grid).
3. **Extract features** (NDVI, NDBI, slope, GHI).
4. **Label & Train** Random Forest; report metrics.
5. **Predict** probabilities over the AOI and **export** a GeoTIFF.
6. **Visualize** the probability map.

---

## Input Data

Project expects the following inputs:

- **Config** (`config.py`):
  - `AOI = [lon_min, lat_min, lon_max, lat_max]`
  - `START_DATE`, `END_DATE`, `CLOUD_PCT`, `EXPORT_SCALE`, paths.
- **Raw rasters** in `data/raw/` (created by the downloader or provided by you):
  - `sentinel_composite.tif` (B2,B4,B8,B11 composite, cloud-masked)
  - `slope.tif` (degrees, from DEM)
  - `ghi.tif` (GHI proxy aggregated to AOI)
- **Labels** for training in `data/processed/labels.csv`:
  - CSV with columns: `row,col,label` where `label` ∈ {0,1}

---

## Output Data

The pipeline writes results to `data/processed/` and `models/`:

- `suitability_map.tif` — float32 probabilities in [0,1], georeferenced
- `features.npy` — stacked feature array (internal)
- `rf_model.joblib` — trained Random Forest model
- Console metrics — classification report (F1/Precision/Recall) and ROC-AUC

---

## Project Structure

```
solar_suitability/
├─ README.md
├─ requirements.txt
├─ config.py
├─ data/
│  ├─ raw/
│  │  ├─ sentinel_composite.tif
│  │  ├─ slope.tif
│  │  └─ ghi.tif
│  └─ processed/
│     ├─ sentinel_composite.tif
│     ├─ slope.tif
│     ├─ ghi.tif
│     ├─ features.npy
│     ├─ labels.csv
│     └─ suitability_map.tif
├─ models/
│  └─ rf_model.joblib
├─ scripts/
│  ├─ download_data.py
│  ├─ preprocess.py
│  ├─ extract_features.py
│  ├─ train_model.py
│  ├─ predict_map.py
│  └─ visualize.py
└─ notebooks/
   └─ exploration.ipynb
```

---

## Commands to Run Each Script

> All commands are meant to be run from the repository root, with your virtual environment **activated**.

### 1) Download data
```sh
python scripts/download_data.py   --aoi "$(python -c 'import json,config; print(json.dumps(config.AOI))')"   --start "$(python -c 'import config; print(config.START_DATE)')"   --end   "$(python -c 'import config; print(config.END_DATE)')"   --cloud "$(python -c 'import config; print(config.CLOUD_PCT)')"   --raw-dir "$(python -c 'import config; print(config.RAW_DIR)')"
```
*If your script doesn't accept CLI flags, omit them — it will read from `config.py`.*

Windows PowerShell (if using flags):
```powershell
python scripts/download_data.py `
  --aoi "$(python -c "import json,config; print(json.dumps(config.AOI))")" `
  --start "$(python -c "import config; print(config.START_DATE)")" `
  --end   "$(python -c "import config; print(config.END_DATE)")" `
  --cloud "$(python -c "import config; print(config.CLOUD_PCT)")" `
  --raw-dir "$(python -c "import config; print(config.RAW_DIR)")"
```

### 2) Preprocess / align
```sh
python scripts/preprocess.py   --raw-dir "$(python -c 'import config; print(config.RAW_DIR)')"   --proc-dir "$(python -c 'import config; print(config.PROC_DIR)')"   --export-scale "$(python -c 'import config; print(config.EXPORT_SCALE)')"
```

### 3) Extract features
```sh
python scripts/extract_features.py   --proc-dir "$(python -c 'import config; print(config.PROC_DIR)')"   --out "$(python -c 'import config; print(config.PROC_DIR)')/features.npy"
```

### 4) Create labels
Create a CSV at `data/processed/labels.csv` with columns:
```csv
row,col,label
123,456,1
789,101,0
```
If you have a sampling helper script, run it here.

### 5) Train model
```sh
python scripts/train_model.py   --proc-dir "$(python -c 'import config; print(config.PROC_DIR)')"   --labels "$(python -c 'import config; print(config.PROC_DIR)')/labels.csv"   --model "$(python -c 'import config; print(config.MODEL_PATH)')"
```

### 6) Predict suitability
```sh
python scripts/predict_map.py   --proc-dir "$(python -c 'import config; print(config.PROC_DIR)')"   --model "$(python -c 'import config; print(config.MODEL_PATH)')"   --out "$(python -c 'import config; print(config.PROC_DIR)')/suitability_map.tif"
```

### 7) Visualize
```sh
python scripts/visualize.py   --map "$(python -c 'import config; print(config.PROC_DIR)')/suitability_map.tif"
```

> If your scripts do **not** yet implement these CLI options, they will still work by reading values from `config.py`. The commands above demonstrate a convenient pattern that can be added with `argparse`.

---

## Notes

- **Indices**:
  - `NDVI = (B8 - B4) / (B8 + B4)`
  - `NDBI = (B11 - B8) / (B11 + B8)` (keep consistent across code)
- **Preprocessing**: ensure a single target CRS, resolution, and transform before stacking. Handle NoData consistently and mask clouds/shadows.
- **Model**: start with `RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)`; report F1 and ROC-AUC; inspect feature importances.
- **Large AOIs**: tile the prediction or use a machine with more RAM.

---

## Troubleshooting

- `earthengine: command not found` → install `earthengine-api` or run `python -m ee.cli.eecli authenticate`.
- `ee.Initialize: no project found` → `earthengine set_project <YOUR_EE_PROJECT_ID>` or set `EARTHENGINE_PROJECT`.
- Process killed during prediction → memory pressure; tile AOI or increase RAM/swap.
- Misaligned rasters → recheck `preprocess.py` target grid and resampling.

---

## License

MIT (adjust if your data sources require specific attribution).

---

## Acknowledgments

Built on open Copernicus Sentinel-2 data, global DEMs, and the Python geospatial ecosystem.
