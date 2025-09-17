# scripts/train_model.py
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import rasterio
from config import PROC_DIR, MODEL_PATH

# incarcam features (flat) si rasterul pt dimensiuni
features = np.load(os.path.join(PROC_DIR, 'features.npy'))  # shape: [n_pixels, n_features]

with rasterio.open(os.path.join(PROC_DIR, 'sentinel_composite.tif')) as src:
    height, width = src.height, src.width

labels_df = pd.read_csv(os.path.join(PROC_DIR, 'labels.csv'))

# transformam (row,col) in index liniar: idx = row * width + col
rows = labels_df['row'].to_numpy()
cols = labels_df['col'].to_numpy()
flat_idx = rows * width + cols

X = features[flat_idx]
y = labels_df['label'].to_numpy().astype(int)

# split si antrenare
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
y_prob = rf.predict_proba(X_val)[:,1]
print(classification_report(y_val, y_pred, target_names=['Unsuitable','Suitable']))
print("Validation AUC:", roc_auc_score(y_val, y_prob))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
joblib.dump(rf, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")
