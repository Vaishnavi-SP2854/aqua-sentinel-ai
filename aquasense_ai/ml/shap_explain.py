"""
shap_explain.py
Generates SHAP explanations for Random Forest predictions.
SHAP = SHapley Additive exPlanations
Answers: "Why did the AI predict HIGH RISK for this water sample?"
Run AFTER train_rf.py
Usage: python shap_explain.py
"""

import os
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# ─── Paths ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
MODELS_DIR    = "models"

# ─── Load everything ──────────────────────────────────────────────────────────
print("Loading model and data...")
rf_contamination = joblib.load(os.path.join(MODELS_DIR, "rf_contamination.pkl"))
rf_risk          = joblib.load(os.path.join(MODELS_DIR, "rf_risk.pkl"))
scaler           = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
le_contamination = joblib.load(os.path.join(MODELS_DIR, "le_contamination.pkl"))
le_risk          = joblib.load(os.path.join(MODELS_DIR, "le_risk.pkl"))
feature_cols     = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))

X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
yc_test = np.load(os.path.join(PROCESSED_DIR, "yc_test.npy"))

print(f"Test samples: {X_test.shape[0]}")
print(f"Features: {feature_cols}")

# ─── Create SHAP Explainer ────────────────────────────────────────────────────
# TreeExplainer is optimized specifically for tree-based models (RF, XGBoost)
# Much faster than KernelExplainer for Random Forests
print("\nBuilding SHAP TreeExplainer...")
explainer = shap.TreeExplainer(rf_contamination)

# Use a subset for speed (SHAP on full test set can be slow)
EXPLAIN_SAMPLES = min(200, X_test.shape[0])
X_explain = X_test[:EXPLAIN_SAMPLES]

print(f"Computing SHAP values for {EXPLAIN_SAMPLES} samples...")
shap_values = explainer.shap_values(X_explain)
# shap_values shape: [n_classes, n_samples, n_features]
print(f"SHAP values shape: {np.array(shap_values).shape}")

# ─── Plot 1: SHAP Summary — Feature Importance (Beeswarm) ───────────────────
# Shows which features matter most AND how (high value → positive/negative impact)
print("\nGenerating SHAP summary plot...")
plt.figure(figsize=(10, 7))
shap.summary_plot(
    shap_values,
    X_explain,
    feature_names=feature_cols,
    class_names=list(le_contamination.classes_),
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance per Contamination Type", pad=12)
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "shap_summary.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: models/shap_summary.png")

# ─── Plot 2: Single Sample Explanation (Waterfall) ───────────────────────────
# This is what gets shown to the citizen after their report is submitted
# "Your water was flagged as HIGH RISK because turbidity=8.2 (↑+0.34) ..."
print("Generating SHAP waterfall plot for a single sample...")

# Find a high-risk sample to explain
high_risk_idx = np.where(rf_contamination.predict(X_explain) != 0)[0]
sample_idx = int(high_risk_idx[0]) if len(high_risk_idx) > 0 else 0
pred_class = rf_contamination.predict(X_explain[sample_idx:sample_idx+1])[0]
pred_name  = le_contamination.classes_[pred_class]

print(f"  Explaining sample {sample_idx}: predicted = {pred_name}")

# Get SHAP values for the predicted class
shap_vals_sample = shap_values[pred_class][sample_idx]
feature_vals_sample = X_explain[sample_idx]

# Sort by absolute SHAP value for clear visualization
sorted_idx = np.argsort(np.abs(shap_vals_sample))[::-1]
top_n = min(8, len(feature_cols))

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#D85A30" if v > 0 else "#185FA5" for v in shap_vals_sample[sorted_idx[:top_n]]]
bars = ax.barh(
    [feature_cols[i] for i in sorted_idx[:top_n]],
    shap_vals_sample[sorted_idx[:top_n]],
    color=colors, edgecolor="white", height=0.6
)
ax.axvline(0, color="gray", linewidth=0.8)
ax.set_title(f"Why AI predicted: {pred_name}\n"
             f"(Red = increases risk, Blue = decreases risk)",
             fontsize=12, pad=10)
ax.set_xlabel("SHAP Value (impact on prediction)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "shap_waterfall_sample.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: models/shap_waterfall_sample.png")

# ─── Save SHAP Explainer for Django API ──────────────────────────────────────
# Django will load this explainer to generate explanations on-the-fly
joblib.dump(explainer, os.path.join(MODELS_DIR, "shap_explainer.pkl"))
print("  Saved: models/shap_explainer.pkl")

# ─── Generate Explanation Text (what Django will show users) ───────────────
def generate_explanation_text(shap_vals, feature_names, feature_values,
                               pred_label, scaler):
    """
    Converts SHAP values into human-readable explanation text.
    This is what the citizen sees on the result page.

    Example output:
      "Your water shows HIGH RISK. The key factors are:
       - Turbidity (8.2 NTU): HIGH — strongly indicates contamination
       - pH (5.1): LOW — outside safe range of 6.5–8.5
       - Organic Carbon (18.4): HIGH — suggests bacterial activity"
    """
    # Un-scale feature values for human-readable display
    original_vals = scaler.inverse_transform(feature_values.reshape(1, -1))[0]

    # Sort by absolute SHAP value (most impactful first)
    sorted_indices = np.argsort(np.abs(shap_vals))[::-1]

    explanation_factors = []
    for i in sorted_indices[:5]:   # Top 5 factors
        feat_name = feature_names[i]
        feat_val  = original_vals[i]
        shap_val  = shap_vals[i]
        impact    = "increases" if shap_val > 0 else "decreases"
        strength  = "strongly " if abs(shap_val) > 0.1 else ""
        explanation_factors.append({
            "feature":      feat_name,
            "value":        round(float(feat_val), 3),
            "shap_value":   round(float(shap_val), 4),
            "impact":       impact,
            "strength":     strength,
            "description":  f"{feat_name} = {feat_val:.2f} — {strength}{impact} risk"
        })

    return {
        "predicted_contamination": pred_label,
        "top_factors": explanation_factors,
        "summary": (
            f"Water classified as {pred_label}. "
            f"Main factors: "
            + ", ".join([f["feature"] for f in explanation_factors[:3]])
        )
    }

# Test the explanation generator
sample_shap = shap_values[pred_class][sample_idx]
explanation = generate_explanation_text(
    sample_shap, feature_cols,
    X_explain[sample_idx], pred_name, scaler
)

print("\nSample explanation output (what users will see):")
print(json.dumps(explanation, indent=2))

# Save the generator function spec as JSON for Django reference
explanation_template = {
    "contamination_types": list(le_contamination.classes_),
    "risk_levels": list(le_risk.classes_),
    "feature_columns": feature_cols,
    "shap_explainer_path": "models/shap_explainer.pkl",
    "explanation_sample": explanation
}
with open(os.path.join(MODELS_DIR, "explanation_template.json"), "w") as f:
    json.dump(explanation_template, f, indent=2)
print("\n  Saved: models/explanation_template.json")
print("\nSHAP explainability complete.")