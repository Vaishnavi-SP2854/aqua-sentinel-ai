"""
config/urls.py
Root URL configuration — wires all app routers together.
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # Django admin panel
    path('admin/', admin.site.urls),

    # API routes — all prefixed with /api/
    path('api/auth/',        include('apps.accounts.urls')),
    path('api/reports/',     include('apps.reports.urls')),
    path('api/predictions/', include('apps.predictions.urls')),
    path('api/maps/',        include('apps.maps.urls')),

    # Frontend template views (Day 3)
    path('', include('apps.reports.urls')),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL,
                          document_root=settings.STATIC_ROOT)