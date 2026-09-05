from rest_framework import serializers
from .models import RiskZone, CityLocation


class RiskZoneSerializer(serializers.ModelSerializer):
    risk_label = serializers.ReadOnlyField()
    map_color  = serializers.ReadOnlyField()

    class Meta:
        model  = RiskZone
        fields = ['id', 'city', 'latitude', 'longitude', 'risk_score',
                  'risk_label', 'map_color', 'dominant_contamination',
                  'report_count', 'last_updated']


class CityLocationSerializer(serializers.ModelSerializer):
    class Meta:
        model  = CityLocation
        fields = '__all__'