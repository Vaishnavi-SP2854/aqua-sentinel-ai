"""
apps/reports/serializers.py
"""

from rest_framework import serializers
from .models import WaterReport, ContaminationType


class ContaminationTypeSerializer(serializers.ModelSerializer):
    class Meta:
        model  = ContaminationType
        fields = '__all__'


class WaterReportSerializer(serializers.ModelSerializer):
    """
    Used for creating a new report (POST).
    user is auto-set from the JWT token — not from request body.
    """
    user       = serializers.StringRelatedField(read_only=True)
    symptoms   = serializers.SerializerMethodField(read_only=True)
    image_url  = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model  = WaterReport
        fields = [
            'id', 'user', 'latitude', 'longitude', 'address', 'city', 'ward',
            'image', 'image_url', 'water_source', 'water_color', 'water_smell',
            'turbidity_score', 'symptoms_diarrhea', 'symptoms_vomiting',
            'symptoms_skin_rash', 'symptoms_fever', 'symptoms_none',
            'ph_value', 'tds_value', 'status', 'symptoms', 'created_at',
        ]
        read_only_fields = ['id', 'user', 'status', 'created_at']

    def get_symptoms(self, obj):
        return obj.symptoms_list

    def get_image_url(self, obj):
        if obj.image:
            request = self.context.get('request')
            return request.build_absolute_uri(obj.image.url) if request else obj.image.url
        return None

    def create(self, validated_data):
        # Auto-attach the logged-in user
        validated_data['user'] = self.context['request'].user
        return super().create(validated_data)


class WaterReportListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for list views — fewer fields."""
    class Meta:
        model  = WaterReport
        fields = ['id', 'city', 'latitude', 'longitude',
                  'water_color', 'status', 'created_at']