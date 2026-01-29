def profile_data(df):
    print('Profiling data...')
    print(df.describe(include='all'))
    print('\nColumn types:')
    print(df.dtypes)
    print('\nMissing values:')
    print(df.isna().sum())

def classify_data(df):
    types = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            types[col] = 'text'
        elif df[col].dtype in ['int64','float64']:
            types[col] = 'numerical'
        else:
            types[col] = 'other'
    print('Column classification:', types)
    return types
