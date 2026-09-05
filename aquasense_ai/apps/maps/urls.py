from django.urls import path
from . import views

urlpatterns = [
    path('risk-zones/', views.RiskZoneListView.as_view(), name='risk-zones'),
    path('cities/',     views.CityListView.as_view(),     name='cities'),
]
