from django.urls import path
from . import views

urlpatterns = [
    # API Endpoints
    path('predict/', views.predict_api, name='predict_api'),
    path('chat/', views.chat_api, name='chat_api'),
]

