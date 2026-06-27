import os
import sys
import pandas as pd
import py_compile

ROOT = os.path.abspath(os.path.dirname(__file__))
print(f'ROOT: {ROOT}')

for path in ['main.py', 'core/router.py', 'engines/diabetes_automl_engine.py', 'core/DIABETES AI SYSTEM.py']:
    full = os.path.join(ROOT, path)
    print(f'Compiling: {full}')
    py_compile.compile(full, doraise=True)

sys.path.insert(0, ROOT)
from core.router import route_to_engines

rows = []
for i in range(50):
    rows.append({
        'Pregnancies': 1,
        'Glucose': 90,
        'BloodPressure': 70,
        'SkinThickness': 20,
        'Insulin': 85,
        'BMI': 24.0,
        'DiabetesPedigreeFunction': 0.25,
        'Age': 22,
        'Outcome': 0,
    })
for i in range(50):
    rows.append({
        'Pregnancies': 3,
        'Glucose': 160,
        'BloodPressure': 85,
        'SkinThickness': 30,
        'Insulin': 180,
        'BMI': 34.0,
        'DiabetesPedigreeFunction': 0.55,
        'Age': 50,
        'Outcome': 1,
    })

df = pd.DataFrame(rows)
print('DATA SHAPE:', df.shape)

output = route_to_engines(df, {col: 'numerical' for col in df.columns})
print('--- SMOKE TEST RESULTS ---')
print('PREDICTIONS =', output.get('predictions'))
print('MODEL_MONITORING_STATUS =', output.get('model_monitoring', {}).get('status'))
print('DIABETES_DETECTION =', output.get('diabetes_detection'))
print('GRAPH_FOLDER =', output.get('graph_folder'))
print('REPORTED TARGETS =', output.get('diabetes_targets'))
