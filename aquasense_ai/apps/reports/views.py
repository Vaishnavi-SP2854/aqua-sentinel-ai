"""
apps/reports/views.py
Report submission and listing API endpoints.
"""

from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from .models import WaterReport
from .serializers import WaterReportSerializer, WaterReportListSerializer
from apps.predictions.tasks import run_prediction_task


class WaterReportCreateView(generics.CreateAPIView):
    """
    POST /api/reports/
    Citizen submits a new water report (photo + form).
    After saving, triggers async Celery prediction task.
    """
    serializer_class   = WaterReportSerializer
    permission_classes = [IsAuthenticated]
    parser_classes     = [MultiPartParser, FormParser, JSONParser]

    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data,
                                          context={'request': request})
        if serializer.is_valid():
            report = serializer.save()

            # Trigger async AI prediction (non-blocking)
            # Celery picks this up in the background — user gets instant response
            run_prediction_task.delay(report.id)

            return Response({
                'message': 'Report submitted. AI analysis in progress.',
                'report':  serializer.data,
                'task':    'prediction queued'
            }, status=status.HTTP_201_CREATED)

        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class WaterReportListView(generics.ListAPIView):
    """
    GET /api/reports/
    Returns logged-in user's own reports.
    """
    serializer_class   = WaterReportListSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return WaterReport.objects.filter(user=self.request.user)


class WaterReportDetailView(generics.RetrieveAPIView):
    """
    GET /api/reports/<id>/
    Returns full detail of one report including prediction result.
    """
    serializer_class   = WaterReportSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        # Users can only view their own reports
        return WaterReport.objects.filter(user=self.request.user)


class PublicReportMapView(generics.ListAPIView):
    """
    GET /api/reports/map/
    Returns all reports (minimal fields) for the public Leaflet map.
    No auth required so the map is publicly visible.
    """
    serializer_class   = WaterReportListSerializer
    permission_classes = [AllowAny]

    def get_queryset(self):
        # Only processed reports show on the map
        qs = WaterReport.objects.filter(status='processed')

        # Optional city filter: /api/reports/map/?city=Nagpur
        city = self.request.query_params.get('city')
        if city:
            qs = qs.filter(city__icontains=city)
        return qs