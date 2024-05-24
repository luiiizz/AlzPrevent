from django.shortcuts import render

# Create your views here.

def home(request):
    return render(request, 'home/home.html')


def questionario(request):
    return render(request, 'questionario/questionario.html')