from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def run_ml(X, y):
    print("🤖 Running ML engine (Step 8)...")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Baseline model (classification)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Evaluation
    acc = accuracy_score(y_test, preds)
    print(f"📊 Model accuracy: {acc:.2f}")

    print("✅ ML training complete")
