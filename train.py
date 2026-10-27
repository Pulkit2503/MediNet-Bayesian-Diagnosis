import pandas as pd
import pickle
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

df = pd.read_csv('patients.csv')

edges = [
    ('Age', 'Hypertension'), ('Smoking', 'Hypertension'), ('Exercise', 'Hypertension'),
    ('Hypertension', 'HeartDisease'), ('FamilyHistory', 'HeartDisease'),
    ('HeartDisease', 'ChestPain'), ('HeartDisease', 'ShortnessBreath'),
    ('Hypertension', 'Fatigue'), ('HeartDisease', 'Fatigue'),
]
model = BayesianNetwork(edges)

model.fit(df, estimator=MaximumLikelihoodEstimator)
print(f"✓ Learned {len(model.get_cpds())} CPTs")

with open('bayes_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✓ Saved bayes_model.pkl")

infer = VariableElimination(model)
ev = {'Age':'Old', 'Smoking':'Yes', 'Exercise':'Sedentary', 'FamilyHistory':'Yes'}
res = infer.query(['HeartDisease'], evidence=ev)
print(f"High-risk patient P(HD=Yes) = {float(res.values[1])*100:.1f}%")