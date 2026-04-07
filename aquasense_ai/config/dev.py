"""
config/dev.py
Development settings — import base + override for local dev.
Set DJANGO_SETTINGS_MODULE=config.dev in your .env
"""

from config.base import *

DEBUG = True

ALLOWED_HOSTS = ['*']

# Show full error pages in dev
INSTALLED_APPS += ['django.contrib.staticfiles']

# Simpler logging in dev
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {'class': 'logging.StreamHandler'},
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}