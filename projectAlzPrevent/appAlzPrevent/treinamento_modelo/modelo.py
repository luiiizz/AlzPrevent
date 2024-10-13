#Dica: Executar no Google Colab e gera o arquivo model_xgb.pkl para ser utilizado no django
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Definir o número de indivíduos
n_individuos = 1000

# Gerar o nome dos indivíduos
individuos = [f"individuo{i}" for i in range(1, n_individuos + 1)]

# Gerar colunas de exposições com valores aleatórios simulando dados reais
np.random.seed(42)  # Para reprodutibilidade

# Educational attainment (years of education): entre 0 e 20 anos
education = np.random.uniform(1, 8, n_individuos)

# Post bronchodilator FEV1 (forçado volume expiratório)
fev1 = np.random.choice([0, 1], n_individuos)

# Iron status biomarkers (iron levels): valor de 20 a 160 (mcg/dL)
iron_levels = np.random.uniform(20, 160, n_individuos)

# Neuroticism: binário, 0 ou 1
neuroticism = np.random.choice([0, 1], n_individuos)

# Family history of Alzheimer's disease: binário, 0 ou 1
family_history = np.random.choice([0, 1], n_individuos)

# General cognitive ability: de 1 a 30, representando desempenho cognitivo
cognitive_performance = np.random.uniform(1, 30, n_individuos)

# Low density lipoprotein cholesterol levels: entre 50 e 200 mg/dL
ldl_cholesterol = np.random.uniform(50, 200, n_individuos)

# Type 1 diabetes: binário, 0 ou 1
type_1_diabetes = np.random.choice([0, 1], n_individuos)

# Parental longevity (Martingale residuals): valores simulados entre 30 e 100
parental_longevity = np.random.uniform(30, 100, n_individuos)

# Weight: valores contínuos simulados entre 50 e 150 kg
weight = np.random.uniform(50, 150, n_individuos)

# Height: valores contínuos simulados entre 1.50 e 2.00 metros
height = np.random.uniform(1.50, 2.00, n_individuos)

# Worry: valores de 0 a 100 representando o grau de preocupação
worry = np.random.uniform(0, 100, n_individuos)

# Diastolic blood pressure: pressão diastólica binário, 0 ou 1
diastolic_bp =  np.random.choice([0, 1], n_individuos)

# Highest math class taken (MTAG): valores simulados entre 1 e 5 (representando nível)
highest_math = np.random.uniform(1, 5, n_individuos)

# Intelligence (MTAG): de 0 a 100, representando desempenho em testes de inteligência
intelligence = np.random.uniform(60, 140, n_individuos)

# Extrinsic epigenetic age acceleration: idade epigenética
epigenetic_age = np.random.normal(42.5, 20, n_individuos)

# Odds ratios (OR) simuladas para cada exposição baseada na tabela
or_education = 0.569570395
or_neuroticism = 1.001279075
or_intelligence = 0.671672587
or_cognitive_performance = 0.998538089
or_highest_math = 0.650177437
or_ldl_cholesterol = 1.906700638
or_family_history = 1.022481281
or_type_1_diabetes = 1.043631596
or_parental_longevity = 0.214121974
or_weight = 0.607929803
or_height = 0.188191974
or_worry = 1.617789409
or_diastolic_bp = 0.942787632
or_fev1 = 1.322015221
or_iron_levels = 0.775180544
or_epigenetic_age = 1.0258077421715

# p-values para ajustar a ponderação das variáveis
p_education = 5.10E-06
p_neuroticism = 0.000507379
p_intelligence = 1.89E-18
p_cognitive_performance = 2.75E-10
p_highest_math = 1.96E-13
p_ldl_cholesterol = 9.56E-06
p_family_history = 9.80E-24
p_type_1_diabetes = 0.042283958
p_parental_longevity = 7.03E-05
p_weight = 0.001354383
p_height = 0.001647619
p_worry = 0.002242555
p_diastolic_bp = 0.00273117
p_fev1 = 5.03E-45
p_iron_levels = 0.000230725
p_epigenetic_age = 0.430623074719789

# Transformar os p-values em pesos (quanto menor o p-value, maior o peso)
def compute_p_value_weight(p_value):
    return -np.log10(p_value)

weights = {
    'education': compute_p_value_weight(p_education),
    'neuroticism': compute_p_value_weight(p_neuroticism),
    'intelligence': compute_p_value_weight(p_intelligence),
    'cognitive_performance': compute_p_value_weight(p_cognitive_performance),
    'highest_math': compute_p_value_weight(p_highest_math),
    'ldl_cholesterol': compute_p_value_weight(p_ldl_cholesterol),
    'family_history': compute_p_value_weight(p_family_history),
    'type_1_diabetes': compute_p_value_weight(p_type_1_diabetes),
    'parental_longevity': compute_p_value_weight(p_parental_longevity),
    'weight': compute_p_value_weight(p_weight),
    'height': compute_p_value_weight(p_height),
    'worry': compute_p_value_weight(p_worry),
    'diastolic_bp': compute_p_value_weight(p_diastolic_bp),
    'fev1': compute_p_value_weight(p_fev1),
    'iron_levels': compute_p_value_weight(p_iron_levels),
    'epigenetic_age': compute_p_value_weight(p_epigenetic_age),
}

# Normalização das variáveis contínuas
scaler = MinMaxScaler()

education_norm = scaler.fit_transform(education.reshape(-1, 1)).flatten()
iron_levels_norm = scaler.fit_transform(iron_levels.reshape(-1, 1)).flatten()
cognitive_performance_norm = scaler.fit_transform(cognitive_performance.reshape(-1, 1)).flatten()
ldl_cholesterol_norm = scaler.fit_transform(ldl_cholesterol.reshape(-1, 1)).flatten()
parental_longevity_norm = scaler.fit_transform(parental_longevity.reshape(-1, 1)).flatten()
weight_norm = scaler.fit_transform(weight.reshape(-1, 1)).flatten()
height_norm = scaler.fit_transform(height.reshape(-1, 1)).flatten()
worry_norm = scaler.fit_transform(worry.reshape(-1, 1)).flatten()
highest_math_norm = scaler.fit_transform(highest_math.reshape(-1, 1)).flatten()
intelligence_norm = scaler.fit_transform(intelligence.reshape(-1, 1)).flatten()
epigenetic_age_norm = scaler.fit_transform(epigenetic_age.reshape(-1, 1)).flatten()

# Cálculo do risco ajustado (utilizando OR e p-value)
risk_score = (
    education_norm * or_education * weights['education'] +
    fev1 * or_fev1 * weights['fev1'] +
    iron_levels_norm * or_iron_levels * weights['iron_levels'] +
    neuroticism * or_neuroticism * weights['neuroticism'] +
    cognitive_performance_norm * or_cognitive_performance * weights['cognitive_performance'] +
    ldl_cholesterol_norm * or_ldl_cholesterol * weights['ldl_cholesterol'] +
    type_1_diabetes * or_type_1_diabetes * weights['type_1_diabetes']  +
    family_history * or_family_history * weights['family_history'] +
    parental_longevity_norm * or_parental_longevity * weights['parental_longevity'] +
    weight_norm * or_weight * weights['weight'] +
    height_norm * or_height * weights['height'] +
    worry_norm * or_worry * weights['worry'] +
    diastolic_bp * or_diastolic_bp * weights['diastolic_bp'] +
    highest_math_norm * or_highest_math * weights['highest_math'] +
    intelligence_norm * or_intelligence * weights['intelligence'] +
    epigenetic_age_norm * or_epigenetic_age * weights['epigenetic_age']
)

# Normalizar o risco para que ele varie entre 0 e 1, representando a probabilidade de DA
risk_score_normalized = (risk_score - np.min(risk_score)) / (np.max(risk_score) - np.min(risk_score))

# Converter o risco em porcentagem de ocorrência de DA (0% a 100%)
probability_da = risk_score_normalized * 100

# Criar o DataFrame final com a coluna de porcentagem de ocorrência de DA
df = pd.DataFrame({
    'Individuo': individuos,
    'Educational_attainment': education,
    'Post_bronchodilator_FEV1': fev1,
    'Iron_status_biomarkers': iron_levels,
    'Neuroticism': neuroticism,
    'Family_history_of_Alzheimers': family_history,
    'Cognitive_performance': cognitive_performance,
    'LDL_cholesterol_levels': ldl_cholesterol,
    'Type_1_diabetes': type_1_diabetes,
    'Parental_longevity': parental_longevity,
    'Weight': weight,
    'Height': height,
    'Worry': worry,
    'Diastolic_blood_pressure': diastolic_bp,
    'Highest_math_class_taken': highest_math,
    'Intelligence': intelligence,
    'Epigenetic_age': epigenetic_age,
    'Risk_score': risk_score,
    'Probability_of_DA (%)': probability_da
})

# Mostrar as primeiras linhas do DataFrame final
#df.head()

#df.to_csv('tcc_exposures.csv', index=False)



#Etapa 2



import warnings
warnings.filterwarnings('ignore')

# Importar bibliotecas necessárias
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, roc_auc_score, mean_absolute_error, r2_score, classification_report, accuracy_score

# Definir a variável alvo (Target) como uma classificação binária (>= 60% chance de DA) para a Regressão Logística
df['Has_DA'] = (df['Probability_of_DA (%)'] >= 60).astype(int)

# Definir as variáveis independentes (features) e as variáveis alvo
X = df[['Educational_attainment', 'Post_bronchodilator_FEV1', 'Iron_status_biomarkers',
        'Neuroticism', 'Family_history_of_Alzheimers', 'Cognitive_performance',
        'LDL_cholesterol_levels', 'Type_1_diabetes',
        'Parental_longevity', 'Weight', 'Height', 'Worry', 'Diastolic_blood_pressure',
        'Highest_math_class_taken', 'Intelligence', 'Epigenetic_age']]

# Variável alvo para regressão logística (classificação binária)
y_log_reg = df['Has_DA']

# Variável alvo para Random Forest e XGBoost (regressão de probabilidade)
y_regression = df['Probability_of_DA (%)']

# Dividir os dados em treino e teste
X_train, X_test, y_train_log_reg, y_test_log_reg = train_test_split(X, y_log_reg, test_size=0.2, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_regression, test_size=0.2, random_state=42)


# 1. Regressão Logística (com variável binária)
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train_log_reg)
y_pred_log_reg = log_reg.predict(X_test)

# Avaliar o modelo de Regressão Logística
print("Regressão Logística (Classificação Binária):")
print(classification_report(y_test_log_reg, y_pred_log_reg))
print(f"Acurácia: {accuracy_score(y_test_log_reg, y_pred_log_reg)}")

# Comparar ROC-AUC para Regressão Logística
roc_log_reg = roc_auc_score(y_test_log_reg, log_reg.predict_proba(X_test)[:, 1])
print(f"\nROC-AUC Regressão Logística: {roc_log_reg}")

# 2. Random Forest (com variável contínua de probabilidade)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_reg, y_train_reg)
y_pred_rf = rf_model.predict(X_test_reg)

# Avaliar o modelo Random Forest
print("\nRandom Forest (Regressão de Probabilidade):")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_reg, y_pred_rf):.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_reg, y_pred_rf):.4f}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_rf):.4f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test_reg, y_pred_rf)):.4f}")

# 3. XGBoost (com variável contínua de probabilidade)
xgb_model = XGBRegressor(use_label_encoder=False, eval_metric='rmse')
xgb_model.fit(X_train_reg, y_train_reg)
y_pred_xgb = xgb_model.predict(X_test_reg)

# Avaliar o modelo XGBoost
print("\nXGBoost (Regressão de Probabilidade):")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_reg, y_pred_xgb):.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_reg, y_pred_xgb):.4f}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_xgb):.4f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test_reg, y_pred_xgb)):.4f}")


# 4. Gradient Boosting (com variável contínua de probabilidade)
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train_reg, y_train_reg)
y_pred_gb = gb_model.predict(X_test_reg)

# Avaliar o modelo Gradient Boosting
print("\nGradient Boosting (Regressão de Probabilidade):")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test_reg, y_pred_gb):.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_reg, y_pred_gb):.4f}")
print(f"R² Score: {r2_score(y_test_reg, y_pred_gb):.4f}")
print(f"Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test_reg, y_pred_gb)):.4f}")


# Exemplo de previsão para novos dados (Random Forest e XGBoost, previsão em probabilidade)
#novo_dado = np.array([[15, 3.0, 80, 30, 1, 85, 140, 1, 85, 70, 170, 40, 90, 15]])
#novo_dado = np.array([[3, 2.5, 1, 4, 1, 7, 130, 0, 2, 70, 1.75, 5, 80, 6, 110]])

novo_dado = np.array([[
    np.random.uniform(1, 8),  # Educational attainment
    np.random.choice([0, 1]),  # Post bronchodilator FEV1
    np.random.uniform(20, 160),  # Iron status biomarkers
    np.random.choice([0, 1]),  # Neuroticism
    np.random.choice([0, 1]),  # Family history of Alzheimer's disease
    np.random.uniform(1, 30),  # Cognitive performance
    np.random.uniform(50, 200),  # LDL cholesterol levels
    np.random.choice([0, 1]),  # Type 1 diabetes
    np.random.uniform(30, 100),  # Parental longevity
    np.random.uniform(50, 150),  # Weight
    np.random.uniform(1.50, 2.00),  # Height
    np.random.uniform(0, 100),  # Worry
    np.random.choice([0, 1]),  # Diastolic blood pressure
    np.random.uniform(1, 5),  # Highest math class taken
    np.random.uniform(60, 140),   # Intelligence
    np.random.normal(42.5, 20) # Epigenetic age
]])

# Definindo as colunas de acordo com as variáveis
colunas = [
    'Educational_attainment',
    'Post_bronchodilator_FEV1',
    'Iron_status_biomarkers',
    'Neuroticism',
    'Family_history_of_Alzheimers',
    'Cognitive_performance',
    'LDL_cholesterol_levels',
    'Type_1_diabetes',
    'Parental_longevity',
    'Weight',
    'Height',
    'Worry',
    'Diastolic_blood_pressure',
    'Highest_math_class_taken',
    'Intelligence',
    'Epigenetic_age'
]

# Criando um DataFrame a partir de novo_dado
df_novo_dado = pd.DataFrame(novo_dado, columns=colunas)

# Exibindo o DataFrame
#print(df_novo_dado)

# Previsão com Regressão Logística (em classificação)
previsao_log_reg = log_reg.predict(novo_dado)
probabilidade_log_reg = log_reg.predict_proba(novo_dado)[:, 1] * 100  # Probabilidade associada

print(f'\nPrevisão Regressão Logística (Classificação Binária, Has_DA): {previsao_log_reg[0]}')
print(f'Probabilidade Regressão Logística de DA (%): {probabilidade_log_reg[0]:.2f}%')

# Previsão com Random Forest (em probabilidade)
previsao_rf = rf_model.predict(novo_dado)
print(f'Previsão Random Forest (Probabilidade de DA %): {previsao_rf[0]:.2f}%')

# Previsão com XGBoost (em probabilidade)
previsao_xgb = xgb_model.predict(novo_dado)
print(f'Previsão XGBoost (Probabilidade de DA %): {previsao_xgb[0]:.2f}%')

# Previsão com Gradient Boosting
previsao_gb = gb_model.predict(novo_dado)
print(f'Previsão Gradient Boosting (Probabilidade de DA %): {previsao_gb[0]:.2f}%')


#df_novo_dado.head()

#Salvar Modelo em pkl
import pickle

with open('model_xgb.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)


