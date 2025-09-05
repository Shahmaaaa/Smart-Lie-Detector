from django.urls import path
from . import views

urlpatterns = [
    # This URL pattern will point to the deception_detection view
    path('analyze/', views.deception_detection),
]
