from django.urls import path
from . import views

urlpatterns = [
    path('<int:report_id>/', views.PredictionDetailView.as_view(),
         name='prediction-detail'),
]