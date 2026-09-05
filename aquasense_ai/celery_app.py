"""
celery_app.py
Celery application setup. Place this in the project root (same level as manage.py).
"""

import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.dev')

app = Celery('aquasense_ai')

# Load Celery config from Django settings (CELERY_* keys)
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-discover tasks.py in all installed apps
app.autodiscover_tasks()


@app.task(bind=True)
def debug_task(self):
    print(f'Request: {self.request!r}')
