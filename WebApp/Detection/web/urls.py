from django.urls import path
from . import views

urlpatterns = [
    path('show_data/', views.my_view),
    path('home/',views.home),
    path('detections/',views.detection_view,name='detection'),
]