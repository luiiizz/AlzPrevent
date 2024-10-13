from django.shortcuts import render

# Create your views here.

# def home(request):
    # return render(request, 'home/home.html')

def questionario(request):
    return render(request, 'questionario/questionario.html')

def questionario2(request):
    return render(request, 'questionario/questionario2.html')

def questionario3(request):
    return render(request, 'questionario/questionario3.html')

def resultado(request):
    return render(request, 'resultado/resultado.html')


from .forms import PredictionForm
import numpy as np
import pickle  # Para carregar o modelo

# Carregar o modelo XGBoost treinado (você deve ter o modelo salvo como um arquivo .pkl)
model_path = 'appAlzPrevent/treinamento_modelo/model_xgb.pkl'
with open(model_path, 'rb') as model_file:
    xgb_model = pickle.load(model_file)

def home(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extrair dados do formulário
            data = [
                form.cleaned_data['educational_attainment'],
                form.cleaned_data['post_bronchodilator_fev1'],
                form.cleaned_data['iron_status_biomarkers'],
                form.cleaned_data['neuroticism'],
                form.cleaned_data['family_history'],
                form.cleaned_data['cognitive_performance'],
                form.cleaned_data['ldl_cholesterol'],
                form.cleaned_data['type_1_diabetes'],
                form.cleaned_data['parental_longevity'],
                form.cleaned_data['weight'],
                form.cleaned_data['height'],
                form.cleaned_data['worry'],
                form.cleaned_data['diastolic_bp'],
                form.cleaned_data['highest_math'],
                form.cleaned_data['intelligence'],
                form.cleaned_data['epigenetic_age']
            ]

            # Converter dados para o formato adequado do modelo
            input_data = np.array([data])

            # Fazer a previsão usando o modelo XGBoost
            predicted_probability = xgb_model.predict(input_data)[0]

            # Exibir o resultado na página de resultados
            return render(request, 'result.html', {'probability': predicted_probability})

    else:
        form = PredictionForm()
    
    return render(request, 'home.html', {'form': form})
