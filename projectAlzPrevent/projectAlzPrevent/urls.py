from django.contrib import admin
from django.urls import path
from appAlzPrevent import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('home/', views.home, name='home'),
    path('home/sobre', views.sobre, name='sobre'),
    path('home/contato', views.contato, name='contato'),
    path('home/prevencao', views.prevencao, name='prevencao'),
    path('questionario/', views.questionario, name='questionario'),
    path('questionario2/', views.questionario2, name='questionario2'),
    path('questionario3/', views.questionario3, name='questionario3'),
    path('processar_dados/', views.processar_dados, name='processar_dados'),
    path('realizar_predicao/', views.realizar_predicao, name='realizar_predicao'),
    path('resultado/', views.resultado, name='resultado'),
    path('teste/', views.teste, name='teste'),

]
