import pandas as pd
import os

def load_data(path):
    if not os.path.exists(path):
        print(f'❌ File not found: {path}')
        return None

    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(path)
    elif ext in ['.xls', '.xlsx']:
        df = pd.read_excel(path)
    elif ext == '.json':
        df = pd.read_json(path)
    else:
        print(f'❌ Unsupported file type: {ext}')
        return None

    print(f'✅ Data loaded: {len(df)} rows')
    return df
