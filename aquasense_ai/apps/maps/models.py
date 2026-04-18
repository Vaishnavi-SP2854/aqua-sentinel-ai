"""
apps/maps/models.py
RiskZone and CityLocation models for the Leaflet.js heatmap.
"""

from django.db import models


class CityLocation(models.Model):
    """Known cities with their coordinates. Pre-seeded in Day 4."""
    name      = models.CharField(max_length=100, unique=True)
    state     = models.CharField(max_length=100)
    latitude  = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    population = models.PositiveIntegerField(default=0)

    class Meta:
        db_table  = 'city_locations'
        ordering  = ['name']

    def __str__(self):
        return f"{self.name}, {self.state}"


class RiskZone(models.Model):
    """
    Aggregated risk score per city.
    Updated in real-time as new reports come in (via Celery task).
    This is what feeds the Leaflet heatmap.
    """

    city       = models.CharField(max_length=100, unique=True)
    latitude   = models.DecimalField(max_digits=9, decimal_places=6)
    longitude  = models.DecimalField(max_digits=9, decimal_places=6)

    risk_score             = models.FloatField(default=0.0)   # 0-3 scale
    dominant_contamination = models.CharField(max_length=20, blank=True)
    report_count           = models.PositiveIntegerField(default=0)

    last_updated = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = 'risk_zones'
        ordering = ['-risk_score']

    def __str__(self):
        return f"{self.city}: risk={self.risk_score}"

    @property
    def risk_label(self):
        if self.risk_score >= 2.5:  return 'High'
        if self.risk_score >= 1.5:  return 'Medium'
        return 'Low'

    @property
    def map_color(self):
        return {'High': '#E24B4A', 'Medium': '#EF9F27',
                'Low': '#639922'}.get(self.risk_label, '#888780')