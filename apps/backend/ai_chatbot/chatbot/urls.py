from django.urls import path
from .views import chat , upload_document

urlpatterns = [
    path('chat/', chat),
    path('upload_document/', upload_document)
]