from django.shortcuts import render

# Create your views here.

def home(request):
    return render(request, 'home/home.html')

def questionario(request):
    return render(request, 'questionario/questionario.html')

def questionario2(request):
    return render(request, 'questionario/questionario2.html')

def questionario3(request):
    return render(request, 'questionario/questionario3.html')

def resultado(request):
    return render(request, 'resultado/resultado.html')