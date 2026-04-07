"""
preprocess.py
Cleans tabular water quality data + prepares image dataset.
Run AFTER download_data.py
Usage: python preprocess.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import cv2

# ─── Paths ────────────────────────────────────────────────────────────────────
RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── PART 1: Tabular Data Preprocessing ──────────────────────────────────────
print("=" * 55)
print("PART 1: Tabular Data Preprocessing")
print("=" * 55)

df = pd.read_csv(os.path.join(RAW_DIR, "water_potability.csv"))
print(f"Loaded dataset: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}\n")

# Step 1A: Fill missing values with median (robust to outliers)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
print("Missing values filled with median.")

# Step 1B: Create contamination_type if not already present
# This maps water quality parameters to contamination categories
if "contamination_type" not in df.columns:
    print("Creating contamination_type labels from parameters...")
    conditions = [
        (df.get("Turbidity", pd.Series([0]*len(df))) < 1) &
        (df.get("ph", pd.Series([7]*len(df))).between(6.5, 8.5)),

        (df.get("Turbidity", pd.Series([0]*len(df))) > 4) &
        (df.get("Organic_carbon", pd.Series([0]*len(df))) > 15),

        (df.get("Chloramines", pd.Series([0]*len(df))) > 10) |
        (df.get("Trihalomethanes", pd.Series([0]*len(df))) > 80),

        (df.get("Hardness", pd.Series([0]*len(df))) > 300),
    ]
    labels_map = [0, 1, 2, 3]
    df["contamination_type"] = np.select(conditions, labels_map, default=4)

# Step 1C: Create risk_level if not present
if "risk_level" not in df.columns:
    df["risk_level"] = np.where(
        df["contamination_type"] == 0, 0,
        np.where(df["contamination_type"].isin([1, 4]), 2, 1)
    )

# Step 1D: Encode string labels to integers
CONTAMINATION_NAMES = {0: "Safe", 1: "Bacterial", 2: "Chemical",
                        3: "Heavy_Metal", 4: "Sewage"}
RISK_NAMES = {0: "Low", 1: "Medium", 2: "High"}

le_contamination = LabelEncoder()
le_risk = LabelEncoder()

df["contamination_encoded"] = le_contamination.fit_transform(df["contamination_type"])
df["risk_encoded"] = le_risk.fit_transform(df["risk_level"])

# Step 1E: Select feature columns for model input
FEATURE_COLS = [c for c in df.columns if c not in
                ["contamination_type", "risk_level",
                 "contamination_encoded", "risk_encoded", "Potability"]]
print(f"Feature columns: {FEATURE_COLS}")

X = df[FEATURE_COLS].values
y_contamination = df["contamination_encoded"].values
y_risk = df["risk_encoded"].values

# Step 1F: Scale features (critical for Random Forest + neural nets)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1G: Train/test split (80/20)
X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
    X_scaled, y_contamination, y_risk,
    test_size=0.2, random_state=42, stratify=y_contamination
)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
print(f"Contamination classes: {list(le_contamination.classes_)}")
print(f"Risk classes: {list(le_risk.classes_)}")

# Step 1H: Save processed tabular data
np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test)
np.save(os.path.join(PROCESSED_DIR, "yc_train.npy"), yc_train)
np.save(os.path.join(PROCESSED_DIR, "yc_test.npy"),  yc_test)
np.save(os.path.join(PROCESSED_DIR, "yr_train.npy"), yr_train)
np.save(os.path.join(PROCESSED_DIR, "yr_test.npy"),  yr_test)

# Step 1I: Save scaler and encoders for use in Django later
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
joblib.dump(le_contamination, os.path.join(MODELS_DIR, "le_contamination.pkl"))
joblib.dump(le_risk, os.path.join(MODELS_DIR, "le_risk.pkl"))
joblib.dump(FEATURE_COLS, os.path.join(MODELS_DIR, "feature_cols.pkl"))

print("\nTabular preprocessing complete. Saved:")
print(f"  {PROCESSED_DIR}/X_train.npy, X_test.npy")
print(f"  {MODELS_DIR}/scaler.pkl, le_contamination.pkl, le_risk.pkl")

# ─── PART 2: Image Data Preprocessing ────────────────────────────────────
print("\n" + "=" * 55)
print("PART 2: Image Dataset Preparation")
print("=" * 55)

IMAGE_DIR       = os.path.join(RAW_DIR, "water_images")
IMG_PROC_DIR    = os.path.join(PROCESSED_DIR, "images")
CATEGORIES      = ["safe", "turbid", "chemical", "sewage", "heavy_metal"]
IMG_SIZE        = (224, 224)   # MobileNetV2 input size
SAMPLES_PER_CAT = 200          # Synthetic samples per category for demo

os.makedirs(IMG_PROC_DIR, exist_ok=True)


def generate_synthetic_water_image(category: str, idx: int) -> np.ndarray:
    """
    Generates a synthetic water image based on category.
    In production, replace this with real photos from citizens.

    Color rules based on real water contamination appearance:
      safe        → clear blue-ish
      turbid      → murky brown/yellow
      chemical    → greenish with bubbles
      sewage      → dark brown/black
      heavy_metal → grey-green metallic
    """
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    np.random.seed(idx * 7 + CATEGORIES.index(category) * 100)

    color_map = {
        "safe":        ([160, 200, 220], 20),   # (base_color, noise)
        "turbid":      ([80, 110, 150],  35),
        "chemical":    ([100, 160, 100], 30),
        "sewage":      ([40, 50, 60],    25),
        "heavy_metal": ([90, 110, 100],  28),
    }

    base_color, noise_level = color_map[category]

    # Fill with base color + noise
    for c in range(3):
        channel = np.full((224, 224), base_color[c], dtype=np.int32)
        channel += np.random.randint(-noise_level, noise_level, (224, 224))
        img[:, :, c] = np.clip(channel, 0, 255).astype(np.uint8)

    # Add category-specific textures
    if category == "turbid":
        # Add murky particles
        for _ in range(80):
            x, y = np.random.randint(10, 214, 2)
            r = np.random.randint(2, 6)
            cv2.circle(img, (x, y), r, (60, 80, 100), -1)

    elif category == "chemical":
        # Add bubble-like circles
        for _ in range(40):
            x, y = np.random.randint(10, 214, 2)
            r = np.random.randint(3, 10)
            cv2.circle(img, (x, y), r, (200, 230, 180), 1)

    elif category == "sewage":
        # Add dark streaks
        for _ in range(15):
            x1, y1 = np.random.randint(0, 224, 2)
            x2, y2 = np.random.randint(0, 224, 2)
            cv2.line(img, (x1, y1), (x2, y2), (20, 25, 30), 2)

    elif category == "heavy_metal":
        # Add metallic sheen effect
        for i in range(0, 224, 8):
            cv2.line(img, (0, i), (224, i), (95, 115, 105), 1)

    return img


# Generate synthetic images for each category
all_images = []
all_labels = []

for cat_idx, category in enumerate(CATEGORIES):
    cat_dir = os.path.join(IMAGE_DIR, category)
    real_images = []

    # Try to load real images if they exist
    if os.path.exists(cat_dir):
        for fname in os.listdir(cat_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                img_path = os.path.join(cat_dir, fname)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, IMG_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    real_images.append(img)

    if real_images:
        print(f"  {category}: {len(real_images)} real images loaded")
        for img in real_images[:SAMPLES_PER_CAT]:
            all_images.append(img)
            all_labels.append(cat_idx)
    else:
        print(f"  {category}: generating {SAMPLES_PER_CAT} synthetic images...")
        for i in range(SAMPLES_PER_CAT):
            img = generate_synthetic_water_image(category, i)
            all_images.append(img)
            all_labels.append(cat_idx)

# Convert to numpy arrays
all_images = np.array(all_images, dtype=np.float32) / 255.0   # Normalize to [0,1]
all_labels = np.array(all_labels, dtype=np.int32)

# Train/test split for images
from sklearn.model_selection import train_test_split as tts
img_X_train, img_X_test, img_y_train, img_y_test = tts(
    all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# Save image arrays
np.save(os.path.join(IMG_PROC_DIR, "img_X_train.npy"), img_X_train)
np.save(os.path.join(IMG_PROC_DIR, "img_X_test.npy"),  img_X_test)
np.save(os.path.join(IMG_PROC_DIR, "img_y_train.npy"), img_y_train)
np.save(os.path.join(IMG_PROC_DIR, "img_y_test.npy"),  img_y_test)
joblib.dump(CATEGORIES, os.path.join(MODELS_DIR, "image_categories.pkl"))

print(f"\nImage preprocessing complete:")
print(f"  Train images: {img_X_train.shape}")
print(f"  Test images:  {img_X_test.shape}")
print(f"  Categories:   {CATEGORIES}")
print("\nAll preprocessing done. Run train_rf.py and train_cnn.py next.")