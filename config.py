# config.py

# AOI definit ca [lon_min, lat_min, lon_max, lat_max]
AOI = [20.26, 43.63, 29.74, 48.27]  # bounding box Romania aprox.

START_DATE = '2023-01-01'
END_DATE   = '2023-12-31'
CLOUD_PCT  = 30

RAW_DIR    = 'data/raw'
PROC_DIR   = 'data/processed'
MODEL_PATH = 'models/rf_model.joblib'

EXPORT_SCALE = 200  # m/pixel

# ID proiect GEE setat direct
EE_PROJECT = "solar-mapping-468709"

# Folder Assets implicit (daca exporti in Assets)
ASSET_FOLDER_NAME = 'solar_outputs'  # va crea: projects/<EE_PROJECT>/assets/solar_outputs
