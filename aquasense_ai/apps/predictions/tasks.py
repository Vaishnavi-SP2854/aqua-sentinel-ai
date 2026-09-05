"""
apps/predictions/tasks.py
Celery async task — runs AI prediction in the background.

Flow:
  1. WaterReport submitted → report saved to DB
  2. run_prediction_task.delay(report.id) called immediately
  3. Celery worker picks up task → runs CNN + RF + SHAP
  4. Saves Prediction to DB
  5. Updates WaterReport status to 'processed'
  6. Frontend polls /api/predictions/<report_id>/ to get result
"""

import os
import sys
import logging
from celery import shared_task
from django.conf import settings

logger = logging.getLogger(__name__)

# Add ml/ directory to Python path so we can import predict.py
ML_DIR = os.path.join(settings.BASE_DIR, 'ml')
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)


@shared_task(bind=True, max_retries=3)
def run_prediction_task(self, report_id: int):
    """
    Main Celery task. Called with .delay() to run asynchronously.
    bind=True lets us access self for retry logic.
    max_retries=3 means it'll retry up to 3 times if it fails.
    """
    from apps.reports.models import WaterReport
    from apps.predictions.models import Prediction, ExplanationLog

    logger.info(f"Starting prediction for report #{report_id}")

    try:
        # ── Load the report ───────────────────────────────────────────────────
        report = WaterReport.objects.get(id=report_id)
        report.status = 'pending'
        report.save(update_fields=['status'])

        # ── Build form data dict from the report model fields ─────────────────
        # These map to the feature_cols expected by predict.py
        form_data = {
            'ph':              report.ph_value    or 7.0,
            'Turbidity':       report.turbidity_score * 2.0,   # Scale 1-5 to ~2-10
            'Hardness':        200.0,   # Default if not provided
            'Solids':          15000.0,
            'Chloramines':     7.0,
            'Sulfate':         330.0,
            'Conductivity':    420.0,
            'Organic_carbon':  14.0,
            'Trihalomethanes': 66.0,
        }

        # Override with actual pH if submitted
        if report.ph_value:
            form_data['ph'] = float(report.ph_value)
        if report.tds_value:
            form_data['Solids'] = float(report.tds_value)

        # Adjust estimates based on smell and color (simple heuristics)
        if report.water_smell == 'sewage':
            form_data['Organic_carbon'] = 22.0
            form_data['Turbidity']     = max(form_data['Turbidity'], 6.0)
        elif report.water_smell == 'chlorine':
            form_data['Chloramines'] = 11.0
        elif report.water_smell == 'sulfur':
            form_data['Sulfate'] = 450.0

        if report.water_color == 'yellow':
            form_data['Turbidity'] = max(form_data['Turbidity'], 5.0)
        elif report.water_color == 'black':
            form_data['Organic_carbon'] = 25.0
            form_data['Turbidity'] = 8.0
        elif report.water_color == 'green':
            form_data['Organic_carbon'] = 20.0

        # Symptom scoring — if people are sick, raise risk estimates
        symptom_count = sum([
            report.symptoms_diarrhea,
            report.symptoms_vomiting,
            report.symptoms_fever,
            report.symptoms_skin_rash,
        ])
        if symptom_count >= 2:
            form_data['Turbidity'] = max(form_data['Turbidity'], 5.0)

        # ── Run prediction ────────────────────────────────────────────────────
        from predict import WaterQualityPredictor
        predictor = WaterQualityPredictor()

        image_path = None
        if report.image:
            image_path = report.image.path

        if image_path and os.path.exists(image_path):
            result = predictor.predict_combined(image_path, form_data)
        else:
            result = predictor.predict_from_form(form_data)
            result['fusion_note'] = 'No image — form data only'
            result['cnn_result']  = None

        logger.info(f"Prediction result for #{report_id}: "
                    f"{result['contamination_type']} / {result['risk_level']}")

        # ── Save Prediction to DB ─────────────────────────────────────────────
        cnn = result.get('cnn_result') or {}
        rf  = result.get('rf_result')  or result

        prediction = Prediction.objects.create(
            report             = report,
            contamination_type = result['contamination_type'],
            risk_level         = result['risk_level'],
            confidence         = result.get('confidence', 0.0),
            fusion_note        = result.get('fusion_note', ''),
            cnn_contamination  = cnn.get('contamination_type', ''),
            cnn_confidence     = cnn.get('confidence', 0.0),
            rf_contamination   = rf.get('contamination_type', ''),
            rf_confidence      = rf.get('contamination_confidence', 0.0),
            recommendations    = result.get('recommendations', []),
            shap_explanation   = result.get('shap_explanation', ''),
        )

        # ── Save SHAP factor details to ExplanationLog ────────────────────────
        for rank, factor in enumerate(result.get('top_factors', []), start=1):
            ExplanationLog.objects.create(
                prediction = prediction,
                feature    = factor['feature'],
                value      = factor['value'],
                shap_value = factor['shap_value'],
                impact     = factor['impact'],
                rank       = rank,
            )

        # ── Update report status ──────────────────────────────────────────────
        report.status = 'processed'
        report.save(update_fields=['status'])

        # ── Update city risk zone ─────────────────────────────────────────────
        update_risk_zone.delay(report.city, result['contamination_type'],
                                result['risk_level'],
                                float(report.latitude),
                                float(report.longitude))

        logger.info(f"Prediction #{prediction.id} saved for report #{report_id}")
        return {'status': 'success', 'prediction_id': prediction.id}

    except WaterReport.DoesNotExist:
        logger.error(f"Report #{report_id} not found")
        return {'status': 'error', 'message': 'Report not found'}

    except Exception as exc:
        logger.error(f"Prediction failed for report #{report_id}: {exc}")
        # Mark report as failed
        try:
            WaterReport.objects.filter(id=report_id).update(status='failed')
        except Exception:
            pass
        # Retry with exponential backoff: 60s, 120s, 240s
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))


@shared_task
def update_risk_zone(city: str, contamination: str,
                     risk_level: str, lat: float, lng: float):
    """
    Updates or creates a RiskZone entry for the city.
    Called after every new prediction to keep the heatmap current.
    """
    from apps.maps.models import RiskZone

    RISK_SCORE = {'Low': 1, 'Medium': 2, 'High': 3}

    zone, created = RiskZone.objects.get_or_create(
        city=city,
        defaults={
            'latitude':          lat,
            'longitude':         lng,
            'risk_score':        RISK_SCORE.get(risk_level, 1),
            'dominant_contamination': contamination,
            'report_count':      1,
        }
    )

    if not created:
        # Update rolling average risk score
        zone.report_count += 1
        new_score = RISK_SCORE.get(risk_level, 1)
        zone.risk_score = round(
            (zone.risk_score * (zone.report_count - 1) + new_score)
            / zone.report_count, 2
        )
        zone.dominant_contamination = contamination
        zone.save(update_fields=['risk_score', 'dominant_contamination',
                                  'report_count'])

    logger.info(f"Risk zone updated: {city} → score {zone.risk_score}")