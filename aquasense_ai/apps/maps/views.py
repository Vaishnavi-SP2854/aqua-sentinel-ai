from rest_framework import generics
from rest_framework.permissions import AllowAny
from .models import RiskZone, CityLocation
from .serializers import RiskZoneSerializer, CityLocationSerializer

class RiskZoneListView(generics.ListAPIView):
    serializer_class   = RiskZoneSerializer
    permission_classes = [AllowAny]
    queryset           = RiskZone.objects.all()

class CityListView(generics.ListAPIView):
    serializer_class   = CityLocationSerializer
    permission_classes = [AllowAny]
    queryset           = CityLocation.objects.all()
