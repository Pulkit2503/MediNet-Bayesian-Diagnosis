import numpy as np
import pandas as pd

np.random.seed(42)
N = 10000

# ROOT NODES
age      = np.random.choice(['Young','Middle','Old'], size=N, p=[0.45, 0.35, 0.20])
smoking  = np.random.choice(['No','Yes'],            size=N, p=[0.70, 0.30])
exercise = np.random.choice(['Regular','Sedentary'], size=N, p=[0.45, 0.55])
family   = np.random.choice(['No','Yes'],            size=N, p=[0.70, 0.30])

# HYPERTENSION
hypertension = []
for a, s, e in zip(age, smoking, exercise):
    p = 0.08
    if a == 'Middle': p += 0.20
    if a == 'Old':    p += 0.38
    if s == 'Yes':    p += 0.15
    if e == 'Sedentary': p += 0.12
    p = min(p, 0.92)
    hypertension.append(np.random.choice(['No','Yes'], p=[1-p, p]))
hypertension = np.array(hypertension)

# HEART DISEASE
heart_disease = []
for h, f in zip(hypertension, family):
    p = 0.04
    if h == 'Yes': p += 0.30
    if f == 'Yes': p += 0.18
    p = min(p, 0.85)
    heart_disease.append(np.random.choice(['No','Yes'], p=[1-p, p]))
heart_disease = np.array(heart_disease)

# SYMPTOMS
chest_pain = [np.random.choice(['No','Yes'], p=[0.35,0.65] if d=='Yes' else [0.85,0.15]) for d in heart_disease]
shortness_breath = [np.random.choice(['No','Yes'], p=[0.45,0.55] if d=='Yes' else [0.85,0.15]) for d in heart_disease]

fatigue = []
for h, d in zip(hypertension, heart_disease):
    p = 0.10
    if h == 'Yes': p += 0.25
    if d == 'Yes': p += 0.35
    p = min(p, 0.90)
    fatigue.append(np.random.choice(['No','Yes'], p=[1-p, p]))

df = pd.DataFrame({
    'Age': age, 'Smoking': smoking, 'Exercise': exercise,
    'FamilyHistory': family, 'Hypertension': hypertension,
    'HeartDisease': heart_disease, 'ChestPain': chest_pain,
    'ShortnessBreath': shortness_breath, 'Fatigue': fatigue,
})

df.to_csv('patients.csv', index=False)
print("Dataset generated! Shape:", df.shape)
print("HeartDisease rate:", f"{df['HeartDisease'].eq('Yes').mean():.2%}")