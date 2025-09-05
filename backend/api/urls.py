# api/urls.py
from django.urls import path
from .views import AnalyzeVideoView

urlpatterns = [
    path('analyze/', AnalyzeVideoView.as_view(), name='analyze_video'),
]