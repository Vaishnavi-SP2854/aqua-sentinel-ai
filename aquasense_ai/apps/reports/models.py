"""
apps/reports/models.py
WaterReport — the core crowdsourced data model.
Each report = one citizen's water sample submission.
"""

from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class ContaminationType(models.Model):
    """Lookup table for contamination types."""
    name        = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True)
    color_code  = models.CharField(max_length=7, default='#FF0000')   # For map

    class Meta:
        db_table = 'contamination_types'

    def __str__(self):
        return self.name


class WaterReport(models.Model):
    """
    One report = one citizen's water sample.
    Contains: photo, location, form answers, and linked prediction.
    """

    STATUS_CHOICES = [
        ('pending',   'Pending'),
        ('processed', 'Processed'),
        ('failed',    'Failed'),
    ]

    COLOR_CHOICES = [
        ('clear',      'Clear'),
        ('yellow',     'Yellow/Brown'),
        ('green',      'Green'),
        ('black',      'Black'),
        ('white',      'Milky White'),
        ('other',      'Other'),
    ]

    SMELL_CHOICES = [
        ('none',       'No smell'),
        ('chlorine',   'Chlorine/Chemical'),
        ('sulfur',     'Rotten Egg / Sulfur'),
        ('sewage',     'Sewage'),
        ('musty',      'Musty/Earthy'),
        ('other',      'Other'),
    ]

    SOURCE_CHOICES = [
        ('tap',        'Tap water'),
        ('borewell',   'Borewell'),
        ('river',      'River/Lake'),
        ('tanker',     'Water tanker'),
        ('well',       'Open well'),
        ('other',      'Other'),
    ]

    # Relationships
    user = models.ForeignKey(User, on_delete=models.CASCADE,
                             related_name='reports')

    # Location
    latitude  = models.DecimalField(max_digits=9, decimal_places=6)
    longitude = models.DecimalField(max_digits=9, decimal_places=6)
    address   = models.TextField(blank=True)
    city      = models.CharField(max_length=100)
    ward      = models.CharField(max_length=100, blank=True)

    # Water photo
    image = models.ImageField(upload_to='water_images/%Y/%m/',
                               null=True, blank=True)

    # Form fields (what the citizen fills in)
    water_source   = models.CharField(max_length=20, choices=SOURCE_CHOICES)
    water_color    = models.CharField(max_length=20, choices=COLOR_CHOICES)
    water_smell    = models.CharField(max_length=20, choices=SMELL_CHOICES)
    turbidity_score = models.IntegerField(default=1,
                        help_text="1=Clear, 5=Very murky")

    # Symptom reports (checkboxes in the form)
    symptoms_diarrhea   = models.BooleanField(default=False)
    symptoms_vomiting   = models.BooleanField(default=False)
    symptoms_skin_rash  = models.BooleanField(default=False)
    symptoms_fever      = models.BooleanField(default=False)
    symptoms_none       = models.BooleanField(default=True)

    # Water parameters (optional — from test kits if available)
    ph_value    = models.FloatField(null=True, blank=True)
    tds_value   = models.FloatField(null=True, blank=True,
                                     help_text="Total Dissolved Solids ppm")

    # Processing status
    status     = models.CharField(max_length=20, choices=STATUS_CHOICES,
                                   default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table  = 'water_reports'
        ordering  = ['-created_at']

    def __str__(self):
        return f"Report #{self.id} by {self.user.email} at {self.city}"

    @property
    def symptoms_list(self):
        """Returns list of reported symptoms."""
        symptoms = []
        if self.symptoms_diarrhea:  symptoms.append('Diarrhea')
        if self.symptoms_vomiting:  symptoms.append('Vomiting')
        if self.symptoms_skin_rash: symptoms.append('Skin rash')
        if self.symptoms_fever:     symptoms.append('Fever')
        return symptoms or ['None']