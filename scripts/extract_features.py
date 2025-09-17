# extract_features.py
import os
import numpy as np
import rasterio
from config import PROC_DIR

# Paths
SENTINEL_P = os.path.join(PROC_DIR, 'sentinel_composite.tif')
SLOPE_P    = os.path.join(PROC_DIR, 'slope.tif')
GHI_P      = os.path.join(PROC_DIR, 'ghi.tif')

# Read stacks
with rasterio.open(SENTINEL_P) as src:
    blue, red, nir, swir = src.read().astype('float32')
    profile = src.profile

slope = rasterio.open(SLOPE_P).read(1).astype('float32')
ghi   = rasterio.open(GHI_P).read(1).astype('float32')

# Compute indices
ndvi = (nir - red) / (nir + red + 1e-6)
ndbi = (swir - nir) / (swir + nir + 1e-6)

# Stack features: [h, w, f]
features = np.stack([ndvi, ndbi, slope, ghi], axis=-1)
h, w, f = features.shape
features_2d = features.reshape(-1, f)

# Save to .npy for training/prediction
os.makedirs(PROC_DIR, exist_ok=True)
np.save(os.path.join(PROC_DIR, 'features.npy'), features_2d)
print(f"Saved feature array: {features_2d.shape} → features.npy")
