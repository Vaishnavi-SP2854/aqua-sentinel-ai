"""
train_rf.py
Trains Random Forest on tabular water quality form data.
Predicts: contamination_type and risk_level
Run AFTER preprocess.py
Usage: python train_rf.py
"""

import os
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score)
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE   # Handles class imbalance

# ─── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# ─── Load preprocessed data ───────────────────────────────────────────────────
print("Loading preprocessed tabular data...")
X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
yc_train = np.load(os.path.join(PROCESSED_DIR, "yc_train.npy"))
yc_test  = np.load(os.path.join(PROCESSED_DIR, "yc_test.npy"))
yr_train = np.load(os.path.join(PROCESSED_DIR, "yr_train.npy"))
yr_test  = np.load(os.path.join(PROCESSED_DIR, "yr_test.npy"))

le_contamination = joblib.load(os.path.join(MODELS_DIR, "le_contamination.pkl"))
le_risk          = joblib.load(os.path.join(MODELS_DIR, "le_risk.pkl"))
feature_cols     = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# ─── Handle Class Imbalance with SMOTE ───────────────────────────────────────
# SMOTE = Synthetic Minority Over-sampling Technique
# Without this, the model learns to always predict the majority class
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_balanced, yc_train_balanced = smote.fit_resample(X_train, yc_train)
_, yr_train_balanced = smote.fit_resample(X_train, yr_train)
print(f"Balanced train size: {X_train_balanced.shape[0]}")

# ─── MODEL 1: Contamination Type Classifier ───────────────────────────────────
# Predicts: Safe / Bacterial / Chemical / Heavy_Metal / Sewage
print("\n" + "=" * 55)
print("Training MODEL 1: Contamination Type Classifier")
print("=" * 55)

rf_contamination = RandomForestClassifier(
    n_estimators=200,        # 200 trees → better accuracy, still fast
    max_depth=15,            # Limit depth to prevent overfitting
    min_samples_split=5,     # Min 5 samples to split a node
    min_samples_leaf=2,      # Min 2 samples at leaf
    class_weight="balanced", # Extra protection against imbalance
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)

rf_contamination.fit(X_train_balanced, yc_train_balanced)

# Evaluate
yc_pred = rf_contamination.predict(X_test)
acc_c = accuracy_score(yc_test, yc_pred)
print(f"\nContamination Type Accuracy: {acc_c:.4f} ({acc_c*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(
    yc_test, yc_pred,
    target_names=list(le_contamination.classes_)
))

# Cross-validation for reliability check
cv_scores = cross_val_score(rf_contamination, X_train_balanced,
                             yc_train_balanced, cv=5, scoring="accuracy")
print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ─── MODEL 2: Risk Level Classifier ──────────────────────────────────────────
# Predicts: Low / Medium / High
print("\n" + "=" * 55)
print("Training MODEL 2: Risk Level Classifier")
print("=" * 55)

rf_risk = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_risk.fit(X_train_balanced, yr_train_balanced)

yr_pred = rf_risk.predict(X_test)
acc_r = accuracy_score(yr_test, yr_pred)
print(f"\nRisk Level Accuracy: {acc_r:.4f} ({acc_r*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(
    yr_test, yr_pred,
    target_names=list(le_risk.classes_)
))

# ─── Feature Importance Plot ──────────────────────────────────────────────────
print("\nGenerating feature importance plot...")
importances = rf_contamination.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Feature Importance — Contamination Type Prediction",
          fontsize=13, pad=15)
bars = plt.barh(
    [feature_cols[i] for i in indices],
    importances[indices],
    color="#534AB7", alpha=0.8, edgecolor="white"
)
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "feature_importance.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: models/feature_importance.png")

# ─── Confusion Matrix ─────────────────────────────────────────────────────────
cm = confusion_matrix(yc_test, yc_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=list(le_contamination.classes_),
            yticklabels=list(le_contamination.classes_))
plt.title("Confusion Matrix — Contamination Type")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix_rf.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: models/confusion_matrix_rf.png")

# ─── Save Models ─────────────────────────────────────────────────────────
joblib.dump(rf_contamination, os.path.join(MODELS_DIR, "rf_contamination.pkl"))
joblib.dump(rf_risk,          os.path.join(MODELS_DIR, "rf_risk.pkl"))

print(f"\nModels saved:")
print(f"  models/rf_contamination.pkl  (Accuracy: {acc_c*100:.2f}%)")
print(f"  models/rf_risk.pkl           (Accuracy: {acc_r*100:.2f}%)")
print("\nRandom Forest training complete.")