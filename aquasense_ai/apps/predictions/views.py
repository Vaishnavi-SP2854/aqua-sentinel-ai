"""
apps/predictions/views.py
Prediction result API — frontend polls this after submitting a report.
"""

from rest_framework import serializers as drf_serializers
from rest_framework import generics
from rest_framework.permissions import IsAuthenticated
from .models import Prediction, ExplanationLog


class ExplanationLogSerializer(drf_serializers.ModelSerializer):
    class Meta:
        model  = ExplanationLog
        fields = ['feature', 'value', 'shap_value', 'impact', 'rank']


class PredictionSerializer(drf_serializers.ModelSerializer):
    explanation_factors = ExplanationLogSerializer(many=True, read_only=True)

    class Meta:
        model  = Prediction
        fields = [
            'id', 'contamination_type', 'risk_level', 'confidence',
            'fusion_note', 'cnn_contamination', 'cnn_confidence',
            'rf_contamination', 'rf_confidence', 'recommendations',
            'shap_explanation', 'explanation_factors', 'created_at',
        ]


class PredictionDetailView(generics.RetrieveAPIView):
    """
    GET /api/predictions/<report_id>/
    Returns prediction result for a given report.
    Frontend polls this every 3 seconds until status = processed.
    """
    serializer_class   = PredictionSerializer
    permission_classes = [IsAuthenticated]

    def get_object(self):
        report_id = self.kwargs['report_id']
        return Prediction.objects.select_related('report').get(
            report__id=report_id,
            report__user=self.request.user
        )