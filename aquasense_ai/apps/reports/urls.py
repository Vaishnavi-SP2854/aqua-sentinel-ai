"""
apps/reports/urls.py
"""

from django.urls import path
from . import views

urlpatterns = [
    path('',        views.WaterReportCreateView.as_view(), name='report-create'),
    path('list/',   views.WaterReportListView.as_view(),   name='report-list'),
    path('<int:pk>/', views.WaterReportDetailView.as_view(), name='report-detail'),
    path('map/',    views.PublicReportMapView.as_view(),   name='report-map'),
]