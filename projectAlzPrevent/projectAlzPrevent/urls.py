from django.contrib import admin
from django.urls import path
from appAlzPrevent import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('home/', views.home, name='home'),
    path('questionario/', views.questionario, name='questionario'),
]
