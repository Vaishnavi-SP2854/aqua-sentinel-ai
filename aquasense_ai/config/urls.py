from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from apps.reports.views import (index_view, report_form_view, result_view,
                                  map_view, dashboard_view,
                                  login_view, register_view)

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/auth/',        include('apps.accounts.urls')),
    path('api/reports/',     include('apps.reports.urls')),
    path('api/predictions/', include('apps.predictions.urls')),
    path('api/maps/',        include('apps.maps.urls')),
    path('',           index_view,       name='index'),
    path('report/',    report_form_view, name='report'),
    path('result/',    result_view,      name='result'),
    path('map/',       map_view,         name='map'),
    path('dashboard/', dashboard_view,   name='dashboard'),
    path('login/',     login_view,       name='login'),
    path('register/',  register_view,    name='register'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL,  document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
