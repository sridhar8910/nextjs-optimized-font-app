from django.urls import path
from .views import FaceDetectionView, ReferenceImageView

urlpatterns = [
    path('detect/', FaceDetectionView.as_view(), name='face-detection'),
    path('set-reference/', ReferenceImageView.as_view(), name='set-reference'),
]
