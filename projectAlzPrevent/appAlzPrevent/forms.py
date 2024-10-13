from django import forms

class PredictionForm(forms.Form):
    educational_attainment = forms.FloatField(label="Years of Education")
    post_bronchodilator_fev1 = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label="Post Bronchodilator FEV1")
    iron_status_biomarkers = forms.FloatField(label="Iron Levels")
    neuroticism = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label="Neuroticism")
    family_history = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label="Family History of Alzheimer's")
    cognitive_performance = forms.FloatField(label="Cognitive Performance")
    ldl_cholesterol = forms.FloatField(label="LDL Cholesterol Levels")
    type_1_diabetes = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label="Type 1 Diabetes")
    parental_longevity = forms.FloatField(label="Parental Longevity")
    weight = forms.FloatField(label="Weight")
    height = forms.FloatField(label="Height")
    worry = forms.FloatField(label="Worry")
    diastolic_bp = forms.ChoiceField(choices=[(0, 'No'), (1, 'Yes')], label="Diastolic Blood Pressure")
    highest_math = forms.FloatField(label="Highest Math Class Taken")
    intelligence = forms.FloatField(label="Intelligence")
    epigenetic_age = forms.FloatField(label="Epigenetic Age")
