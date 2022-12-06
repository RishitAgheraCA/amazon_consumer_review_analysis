from django.urls import path
from scraping import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.InferenceView.as_view(), name='scraping'),
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)