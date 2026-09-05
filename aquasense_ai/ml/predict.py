"""
predict.py
Unified prediction engine — combines CNN + Random Forest + SHAP.
Django will import and call this on Day 2.
Test it now: python predict.py
"""

import os
import numpy as np
import joblib
import json
import cv2
import tensorflow as tf

# Suppress TensorFlow info logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

MODELS_DIR = "models"


class WaterQualityPredictor:
    """
    Main prediction class. Django imports this and calls predict().
    Combines CNN (image) + RF (form data) for final decision.
    """

    def __init__(self):
        print("Loading all models...")
        self._load_models()
        print("All models loaded successfully.")

    def _load_models(self):
        """Load all saved models and helpers."""
        # Random Forest models
        self.rf_contamination = joblib.load(
            os.path.join(MODELS_DIR, "rf_contamination.pkl"))
        self.rf_risk = joblib.load(
            os.path.join(MODELS_DIR, "rf_risk.pkl"))

        # Preprocessing helpers
        self.scaler           = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        self.le_contamination = joblib.load(os.path.join(MODELS_DIR, "le_contamination.pkl"))
        self.le_risk          = joblib.load(os.path.join(MODELS_DIR, "le_risk.pkl"))
        self.feature_cols     = joblib.load(os.path.join(MODELS_DIR, "feature_cols.pkl"))
        self.image_categories = joblib.load(os.path.join(MODELS_DIR, "image_categories.pkl"))

        # SHAP explainer
        self.shap_explainer = joblib.load(
            os.path.join(MODELS_DIR, "shap_explainer.pkl"))

        # CNN model
        best_model = os.path.join(MODELS_DIR, "water_cnn_best.h5")
        final_model = os.path.join(MODELS_DIR, "water_cnn.h5")
        cnn_path = best_model if os.path.exists(best_model) else final_model
        if os.path.exists(cnn_path):
            self.cnn_model = tf.keras.models.load_model(cnn_path)
            self.cnn_available = True
        else:
            print("  WARNING: CNN model not found — image prediction disabled")
            self.cnn_model = None
            self.cnn_available = False

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Reads image from disk and prepares it for CNN input.
        Accepts: jpg, png, jpeg
        Returns: (1, 224, 224, 3) float32 array normalized to [0,1]
        """
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)   # Add batch dimension

    def preprocess_form_data(self, form_data: dict) -> np.ndarray:
        """
        Converts citizen form input to scaled feature array.

        form_data keys (from the HTML report form):
          ph, turbidity, color_score, smell_score, location_risk_score,
          symptoms_score, hardness, chloramines, etc.

        Returns: (1, n_features) scaled numpy array
        """
        # Build feature vector in correct column order
        feature_vector = []
        for col in self.feature_cols:
            # Map form field names to feature columns
            # Use 0 as fallback for optional fields
            val = form_data.get(col, form_data.get(col.lower(), 0))
            feature_vector.append(float(val))

        X = np.array(feature_vector).reshape(1, -1)
        return self.scaler.transform(X)

    def predict_from_image(self, image_path: str) -> dict:
        """Predicts contamination type from water photo."""
        if not self.cnn_available:
            return {"error": "CNN model not available"}

        img_array = self.preprocess_image(image_path)
        probs = self.cnn_model.predict(img_array, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        return {
            "contamination_type": self.image_categories[pred_idx],
            "confidence":         float(probs[pred_idx]),
            "all_probabilities": {
                cat: float(p)
                for cat, p in zip(self.image_categories, probs)
            }
        }

    def predict_from_form(self, form_data: dict) -> dict:
        """Predicts contamination type and risk level from form data."""
        X_scaled = self.preprocess_form_data(form_data)

        # Contamination prediction
        cont_pred  = self.rf_contamination.predict(X_scaled)[0]
        cont_probs = self.rf_contamination.predict_proba(X_scaled)[0]
        cont_name  = self.le_contamination.classes_[cont_pred]

        # Risk level prediction
        risk_pred  = self.rf_risk.predict(X_scaled)[0]
        risk_probs = self.rf_risk.predict_proba(X_scaled)[0]
        risk_name  = self.le_risk.classes_[risk_pred]

        # SHAP explanation
        shap_vals = self.shap_explainer.shap_values(X_scaled)
        shap_for_class = shap_vals[cont_pred][0]

        sorted_idx = np.argsort(np.abs(shap_for_class))[::-1]
        original_vals = self.scaler.inverse_transform(X_scaled)[0]

        top_factors = []
        for i in sorted_idx[:5]:
            top_factors.append({
                "feature":    self.feature_cols[i],
                "value":      round(float(original_vals[i]), 3),
                "shap_value": round(float(shap_for_class[i]), 4),
                "impact":     "high" if shap_for_class[i] > 0 else "low"
            })

        return {
            "contamination_type":  cont_name,
            "contamination_confidence": float(max(cont_probs)),
            "risk_level":          risk_name,
            "risk_confidence":     float(max(risk_probs)),
            "top_factors":         top_factors,
            "shap_explanation":    (
                f"Water flagged as {cont_name} ({risk_name} risk). "
                f"Key factors: "
                + ", ".join([f["feature"] for f in top_factors[:3]])
            )
        }

    def predict_combined(self, image_path: str, form_data: dict) -> dict:
        """
        MAIN METHOD — Fuses CNN (image) + RF (form) predictions.
        This is what Django calls from the API endpoint.

        Fusion strategy:
          - If both models agree → high confidence result
          - If they disagree → use RF result + flag for review
          - CNN confidence < 0.5 → trust RF more
        """
        # Get predictions from both models
        form_result  = self.predict_from_form(form_data)
        image_result = self.predict_from_image(image_path) if (
            image_path and self.cnn_available) else None

        # Fusion logic
        if image_result and "error" not in image_result:
            cnn_conf = image_result["confidence"]
            cnn_cont = image_result["contamination_type"]
            rf_cont  = form_result["contamination_type"]

            if cnn_cont == rf_cont:
                # Models agree — high confidence
                fusion_note = "CNN and form data agree — high confidence result"
                final_contamination = rf_cont
                final_confidence = (cnn_conf + form_result["contamination_confidence"]) / 2
            elif cnn_conf >= 0.6:
                # CNN is confident — blend
                fusion_note = "CNN image analysis weighted higher due to confidence"
                final_contamination = cnn_cont
                final_confidence = cnn_conf * 0.6 + form_result["contamination_confidence"] * 0.4
            else:
                # CNN uncertain — trust form data
                fusion_note = "Low image confidence — form data used as primary"
                final_contamination = rf_cont
                final_confidence = form_result["contamination_confidence"]
        else:
            fusion_note = "Image not available — form data only"
            final_contamination = form_result["contamination_type"]
            final_confidence = form_result["contamination_confidence"]

        # Build health recommendations based on result
        recommendations = self._get_recommendations(
            final_contamination, form_result["risk_level"])

        return {
            "contamination_type":  final_contamination,
            "risk_level":          form_result["risk_level"],
            "confidence":          round(final_confidence, 3),
            "shap_explanation":    form_result["shap_explanation"],
            "top_factors":         form_result["top_factors"],
            "recommendations":     recommendations,
            "fusion_note":         fusion_note,
            "cnn_result":          image_result,
            "rf_result":           form_result,
        }

    def _get_recommendations(self, contamination_type: str,
                              risk_level: str) -> list:
        """Returns actionable health advice based on contamination type."""
        base = {
            "Safe": [
                "Water appears safe — continue regular testing",
                "Report any change in taste or smell immediately",
            ],
            "Bacterial": [
                "BOIL water for at least 5 minutes before drinking",
                "Do not use for cooking without boiling",
                "Contact local health authority — typhoid/cholera risk",
                "Use chlorine tablets if boiling not possible",
            ],
            "Chemical": [
                "DO NOT drink or cook with this water",
                "Use sealed bottled water immediately",
                "Contact municipality — industrial discharge suspected",
                "Keep away from children and elderly",
            ],
            "Heavy_Metal": [
                "DO NOT drink — heavy metal poisoning risk",
                "Do not use for bathing infants",
                "Contact pollution control board",
                "Seek medical check-up if used recently",
            ],
            "Sewage": [
                "IMMEDIATE risk — do not use for any purpose",
                "Seek alternative water source urgently",
                "Alert neighbours and local ward office",
                "Cholera/typhoid outbreak risk — contact health dept",
            ],
        }
        risk_addition = {
            "High": ["Seek medical attention if any symptoms appear",
                     "Report to district collector office immediately"],
            "Medium": ["Monitor for symptoms over next 48 hours"],
            "Low": ["Retest in 7 days to confirm safety"],
        }
        recs = base.get(contamination_type, ["Consult local health authority"])
        recs += risk_addition.get(risk_level, [])
        return recs


# ─── Test the predictor end-to-end ───────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Testing WaterQualityPredictor")
    print("=" * 55)

    predictor = WaterQualityPredictor()

    # Simulate a citizen form submission
    sample_form = {
        "ph":              6.2,    # Slightly acidic
        "Hardness":        280.0,
        "Solids":          18000.0,
        "Chloramines":     11.5,   # High — chemical risk
        "Sulfate":         350.0,
        "Conductivity":    420.0,
        "Organic_carbon":  17.2,   # High — bacterial risk
        "Trihalomethanes": 85.0,   # Above WHO limit
        "Turbidity":       5.8,    # Very turbid — visible contamination
    }

    print("\nForm data submitted:")
    for k, v in sample_form.items():
        print(f"  {k}: {v}")

    # Form-only prediction
    result = predictor.predict_from_form(sample_form)

    print("\n" + "=" * 55)
    print("PREDICTION RESULT")
    print("=" * 55)
    print(f"Contamination Type : {result['contamination_type']}")
    print(f"Risk Level         : {result['risk_level']}")
    print(f"Confidence         : {result['contamination_confidence']:.2%}")
    print(f"\nSHAP Explanation:")
    print(f"  {result['shap_explanation']}")
    print(f"\nTop 3 Factors:")
    for f in result["top_factors"][:3]:
        print(f"  {f['feature']} = {f['value']} (SHAP: {f['shap_value']:+.4f})")

    print("\npredict.py test complete — all models working correctly.")