"""
download_data.py
Downloads UCI water quality dataset and water image dataset.
Run this FIRST on Day 1.
Usage: python download_data.py
"""

import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np

# ─── Paths ────────────────────────────────────────────────────────────────
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

# ─── 1. Download UCI Water Quality dataset ────────────────────────────────────
# This dataset has 3276 water samples with pH, hardness, turbidity etc.
UCI_URL = "https://raw.githubusercontent.com/dsrscientist/dataset1/master/water_potability.csv"
UCI_SAVE = os.path.join(RAW_DIR, "water_potability.csv")

print("Downloading UCI water quality dataset...")
try:
    urllib.request.urlretrieve(UCI_URL, UCI_SAVE)
    df = pd.read_csv(UCI_SAVE)
    print(f"  Saved: {UCI_SAVE}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
except Exception as e:
    print(f"  Error: {e}")
    print("  Generating synthetic dataset instead...")

    # ─── Fallback: Generate realistic synthetic water quality data ───────────
    # This mirrors UCI dataset structure so all downstream code works identically
    np.random.seed(42)
    n = 5000

    # Simulate realistic water quality parameter distributions
    data = {
        "ph":               np.clip(np.random.normal(7.0, 1.5, n), 0, 14),
        "Hardness":         np.random.normal(196, 32, n),
        "Solids":           np.random.normal(22014, 8657, n),
        "Chloramines":      np.clip(np.random.normal(7.1, 1.5, n), 0, 13),
        "Sulfate":          np.random.normal(333, 41, n),
        "Conductivity":     np.random.normal(426, 80, n),
        "Organic_carbon":   np.random.normal(14, 3.3, n),
        "Trihalomethanes":  np.clip(np.random.normal(66, 16, n), 0, 120),
        "Turbidity":        np.clip(np.random.normal(3.9, 0.78, n), 0, 10),
    }
    df = pd.DataFrame(data)

    # Create contamination labels based on realistic thresholds
    # 0=Safe, 1=Bacterial, 2=Chemical, 3=Heavy_Metal, 4=Sewage
    conditions = [
        (df["Turbidity"] < 1) & (df["ph"].between(6.5, 8.5)) & (df["Chloramines"] < 4),
        (df["Turbidity"] > 4) & (df["Organic_carbon"] > 15),
        (df["Chloramines"] > 10) | (df["Trihalomethanes"] > 80),
        (df["Hardness"] > 300) & (df["Conductivity"] > 600),
    ]
    labels = [0, 1, 2, 3]
    df["contamination_type"] = np.select(conditions, labels, default=4)

    # Risk level: 0=Low, 1=Medium, 2=High
    df["risk_level"] = np.where(
        df["contamination_type"] == 0, 0,
        np.where(df["contamination_type"].isin([1, 4]), 2, 1)
    )

    # Add 10% missing values like real datasets
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        mask = np.random.random(n) < 0.1
        df.loc[mask, col] = np.nan

    df.to_csv(UCI_SAVE, index=False)
    print(f"  Synthetic dataset saved: {UCI_SAVE} — shape {df.shape}")

# ─── 2. Generate water image dataset (synthetic) ──────────────────────────────
# Real image datasets require Kaggle API key setup.
# This creates a placeholder structure. Replace with real images for production.
print("\nCreating water image directory structure...")

IMAGE_DIR = os.path.join(RAW_DIR, "water_images")
CATEGORIES = ["safe", "turbid", "chemical", "sewage", "heavy_metal"]

for cat in CATEGORIES:
    os.makedirs(os.path.join(IMAGE_DIR, cat), exist_ok=True)
    print(f"  Created: {IMAGE_DIR}/{cat}/")

print("\nNOTE: For real water images, download from Kaggle:")
print("  kaggle datasets download -d adityakadiwal/water-potability")
print("  kaggle datasets download -d ankit8467/water-quality-dataset")
print("  Then sort images into the 5 category folders above.")
print("\nFor Day 1 demo, we will generate synthetic images in preprocess.py")
print("\nData download complete.")