"""
apps/predictions/models.py
Prediction and ExplanationLog models.
One WaterReport → one Prediction → one ExplanationLog.
"""

from django.db import models
from apps.reports.models import WaterReport


class Prediction(models.Model):
    """
    AI prediction result linked to a WaterReport.
    Stores both CNN (image) and RF (form) results + fused final decision.
    """

    RISK_CHOICES = [
        ('Low',    'Low'),
        ('Medium', 'Medium'),
        ('High',   'High'),
    ]

    CONTAMINATION_CHOICES = [
        ('Safe',         'Safe'),
        ('Bacterial',    'Bacterial'),
        ('Chemical',     'Chemical'),
        ('Heavy_Metal',  'Heavy Metal'),
        ('Sewage',       'Sewage'),
    ]

    report = models.OneToOneField(WaterReport, on_delete=models.CASCADE,
                                   related_name='prediction')

    # Final fused prediction
    contamination_type = models.CharField(max_length=20,
                                           choices=CONTAMINATION_CHOICES)
    risk_level         = models.CharField(max_length=10, choices=RISK_CHOICES)
    confidence         = models.FloatField(default=0.0)
    fusion_note        = models.TextField(blank=True)

    # CNN-only result
    cnn_contamination  = models.CharField(max_length=20, blank=True)
    cnn_confidence     = models.FloatField(default=0.0)

    # RF-only result
    rf_contamination   = models.CharField(max_length=20, blank=True)
    rf_confidence      = models.FloatField(default=0.0)

    # Health recommendations (stored as JSON list)
    recommendations    = models.JSONField(default=list)

    # SHAP summary text
    shap_explanation   = models.TextField(blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'predictions'

    def __str__(self):
        return (f"Prediction for Report #{self.report.id}: "
                f"{self.contamination_type} ({self.risk_level})")


class ExplanationLog(models.Model):
    """
    Stores full SHAP factor details for one prediction.
    One row per feature factor.
    """

    prediction  = models.ForeignKey(Prediction, on_delete=models.CASCADE,
                                     related_name='explanation_factors')
    feature     = models.CharField(max_length=100)
    value       = models.FloatField()
    shap_value  = models.FloatField()
    impact      = models.CharField(max_length=10)   # 'high' or 'low'
    rank        = models.IntegerField()              # 1 = most important

    class Meta:
        db_table  = 'explanation_logs'
        ordering  = ['rank']

    def __str__(self):
        return f"{self.feature} = {self.value} (SHAP: {self.shap_value:+.4f})"