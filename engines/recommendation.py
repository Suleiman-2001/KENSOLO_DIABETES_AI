def run_recommendations(df):
    recommendations = {}
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if not numeric_cols:
        return {}
    for col in numeric_cols:
        mean_val = df[col].mean()
        recommendations[col] = []
        for i, val in enumerate(df[col]):
            if val < mean_val * 0.8:
                recommendations[col].append({'row': i+1, 'recommendation': 'Needs support', 'value': val})
            elif val > mean_val * 1.2:
                recommendations[col].append({'row': i+1, 'recommendation': 'Excelling', 'value': val})
            else:
                recommendations[col].append({'row': i+1, 'recommendation': 'Average', 'value': val})
    print("Recommendations generated")
    return recommendations
