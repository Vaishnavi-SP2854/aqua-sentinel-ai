"""
apps/accounts/models.py
Custom User model extending Django's AbstractUser.
Adds city, phone number, and profile fields for AquaSense.
"""

from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """
    Extended user model.
    Django's AbstractUser already gives us:
      username, email, password, first_name, last_name, is_active, date_joined
    We add city-specific fields below.
    """

    email = models.EmailField(unique=True)   # Make email unique (login field)
    phone = models.CharField(max_length=15, blank=True)
    city  = models.CharField(max_length=100, blank=True)
    state = models.CharField(max_length=100, blank=True)

    # Location coordinates (for map pinning)
    latitude  = models.DecimalField(max_digits=9,  decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9,  decimal_places=6, null=True, blank=True)

    # Stats
    total_reports    = models.PositiveIntegerField(default=0)
    verified_citizen = models.BooleanField(default=False)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    USERNAME_FIELD  = 'email'        # Login with email, not username
    REQUIRED_FIELDS = ['username']

    class Meta:
        db_table  = 'users'
        ordering  = ['-date_joined']
        verbose_name = 'User'
        verbose_name_plural = 'Users'

    def __str__(self):
        return f"{self.email} ({self.city})"