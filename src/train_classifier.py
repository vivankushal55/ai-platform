import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# Load labelled data
df = pd.read_csv("data/tickets.csv")
X = df["instruction"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Settings we might tune (log these so we can compare runs) ---
MAX_FEATURES = 5000          # MLflow
MODEL_C = 1.0                # MLflow (regularization strength)

mlflow.set_experiment("ticket-classifier")  # MLflow: names the project

with mlflow.start_run():     # MLflow: begins a tracked run
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=MAX_FEATURES)),
        ("clf", LogisticRegression(max_iter=1000, C=MODEL_C)),
    ])

    print("Training...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.1%}\n")
    print(classification_report(y_test, preds))

    # --- MLflow: log the parameters and the metric ---
    mlflow.log_param("max_features", MAX_FEATURES)
    mlflow.log_param("model_C", MODEL_C)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)

    # --- MLflow: save the trained model into the run ---
    mlflow.sklearn.log_model(pipeline, "model")

    # Also keep the local copy for the agent to load later
    joblib.dump(pipeline, "data/classifier.joblib")
    print("Saved model to data/classifier.joblib")

    print(f"\nMLflow logged this run. Accuracy={acc:.1%}")
