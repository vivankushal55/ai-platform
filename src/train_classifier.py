import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

# ---------------------------------------------------------------
# STEP 1: Load the labelled data
# instruction = the ticket text, category = the label to predict
# ---------------------------------------------------------------
df = pd.read_csv("data/tickets.csv")
X = df["instruction"]
y = df["category"]

# ---------------------------------------------------------------
# STEP 2: Split into train/test so we can measure honestly.
# The model learns on 80%, we grade it on the unseen 20%.
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# STEP 3: Build a pipeline:
#   TfidfVectorizer = turns text into numbers by word importance
#   LogisticRegression = a fast, solid classifier
# A Pipeline chains them so .fit() does both steps together.
# ---------------------------------------------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000)),
])

# ---------------------------------------------------------------
# STEP 4: Train
# ---------------------------------------------------------------
print("Training...")
pipeline.fit(X_train, y_train)

# ---------------------------------------------------------------
# STEP 5: Evaluate on the unseen test set
# ---------------------------------------------------------------
preds = pipeline.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"\nAccuracy: {acc:.1%}\n")
print(classification_report(y_test, preds))

# ---------------------------------------------------------------
# STEP 6: Save the trained model to reuse later (in the agent)
# ---------------------------------------------------------------
joblib.dump(pipeline, "data/classifier.joblib")
print("Saved model to data/classifier.joblib")

# ---------------------------------------------------------------
# STEP 7: Quick sanity test on a made-up ticket
# ---------------------------------------------------------------
test = ["I want my money back for this broken item"]
print(f"\nTest: {test[0]}")
print(f"Predicted category: {pipeline.predict(test)[0]}")