from django.shortcuts import render, redirect
from django.http import JsonResponse
import numpy as np
from django.contrib import messages

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


from .forms import PredictionForm
import numpy as np
import pickle  # Para carregar o modelo

# Carregar o modelo XGBoost treinado (você deve ter o modelo salvo como um arquivo .pkl)
model_path = 'appAlzPrevent/treinamento_modelo/model_xgb.pkl'
with open(model_path, 'rb') as model_file:
    xgb_model = pickle.load(model_file)

# Função para testar resposta do modelo
def teste(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Extrair dados do formulário
            data = [
                float(form.cleaned_data['educational_attainment']),
                #int(form.cleaned_data['post_bronchodilator_fev1']),
                float(form.cleaned_data['iron_status_biomarkers']),
                int(form.cleaned_data['neuroticism']),
                int(form.cleaned_data['family_history']),
                float(form.cleaned_data['cognitive_performance']),
                float(form.cleaned_data['ldl_cholesterol']),
                int(form.cleaned_data['type_1_diabetes']),
                float(form.cleaned_data['parental_longevity']),
                float(form.cleaned_data['weight']),
                float(form.cleaned_data['height']),
                float(form.cleaned_data['worry']),
                int(form.cleaned_data['diastolic_bp']),
                float(form.cleaned_data['highest_math']),
                float(form.cleaned_data['intelligence']),
                float(form.cleaned_data['epigenetic_age'])
            ]

            # Converter dados para o formato adequado do modelo
            input_data = np.array([data])

            print('teste')
            print(data)
            print(input_data)

            # Fazer a previsão usando o modelo XGBoost
            predicted_probability = xgb_model.predict(input_data)[0]

            # Exibir o resultado na página de resultados
            return render(request, 'result.html', {'probability': predicted_probability})

    else:
        form = PredictionForm()
    
    return render(request, 'teste.html', {'form': form})


# Função para processar dados de cada questionário
def processar_dados(request):
    if request.method == "POST":
        # Captura dados dos formulários
        dados = request.POST.dict()

        # Acumular os dados dos três questionários em uma sessão
        questionarios = request.session.get('questionarios', {})
        questionarios.update(dados)
        request.session['questionarios'] = questionarios
        #print(questionarios)
        #print(questionarios['altura'])
        #print(questionarios['parental'])

        # Verifique se estamos no último formulário e redirecione para a view de predição
        if 'preocupacao' in dados: # Preencheu todos os questionarios
            return redirect('realizar_predicao')  # Redireciona para a função de predição
            #print('processando')
        
        elif 'colesterol' in dados:  # Preencheu até o segundo questionario
            return redirect('questionario3')  # Redireciona para a função de predição
        
        else: # Preencheu somente o primeiro questionario
            # Redireciona para o próximo questionário
            return redirect('questionario2')  # Mapeie corretamente na URL

    return render(request, 'questionario.html')


def realizar_predicao(request):

    # Pega os dados completos dos três questionários salvos na sessão
    questionarios = request.session.get('questionarios', {})

    print('Questionario submetido:')
    print(questionarios, len(questionarios))

    # Certifique-se de que todos os dados necessários estão presentes
    expected_number_of_fields = 16
    if len(questionarios) < expected_number_of_fields:  # Ajuste para o número de campos necessários
        messages.error(request, 'Por favor, preencha todos os campos antes de continuar.')
        return redirect('questionario')  # Redireciona para o início se os dados estiverem incompletos

    # Remove o campo desnecessário (token CSRF)
    questionarios_limpo = [

        float(questionarios.get('escolaridade', 0)),
        #int(form.cleaned_data['post_bronchodilator_fev1']),
        float(questionarios.get('ferro', 0)),
        int(questionarios.get('neurociticismo', 0)),
        int(questionarios.get('historico', 0)),
        float(questionarios.get('MEEM', 0) or 0), 
        float(questionarios.get('colesterol', 0)),
        int(questionarios.get('diabetes', 0)),
        float(questionarios.get('parental', 0)),
        float(questionarios.get('peso', 0)),
        float(questionarios.get('altura', 0)),
        float(questionarios.get('preocupacao', 0)),
        int(questionarios.get('pressao', 0)),
        float(questionarios.get('matamatica', 0)),
        float(questionarios.get('inteligente', 0)),
        float(questionarios.get('idade', 0))
    ]
    
    # Carrega o modelo e faz a predição
    
    # Converter dados para o formato adequado do modelo
    input_data = np.array([questionarios_limpo])
    
    print('Dados limpos:', input_data)
    resultado = xgb_model.predict(input_data)[0]  # `predict` usa o modelo para prever

    print('Resultado:', resultado)

    # Limpa os dados da sessão após o uso
    #del request.session['questionarios']

    # Renderiza o resultado
    return render(request, 'resultado/resultado.html', {'resultado': resultado})