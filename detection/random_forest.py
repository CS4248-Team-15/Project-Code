import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from data_utils import load_and_prepare_data

def evaluate_model_validation(model, vectorizer, X_val, y_val):
    """Evaluates a trained model on validation and test sets."""
    X_val_tfidf = vectorizer.transform(X_val)

    y_val_pred = model.predict(X_val_tfidf)

    print("Random Forest Validation Set Performance:")
    print(classification_report(y_val, y_val_pred))

def evaluate_model_test(model, X_test, y_test):
    with open("./models/random_forest_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
        X_test_tfidf = vectorizer.transform(X_test)
    y_test_pred = model.predict(X_test_tfidf)
    print("Random Forest Test Set Performance:")
    print(classification_report(y_test, y_test_pred))

# Load and prepare data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data("./data/Sarcasm_Headlines_Dataset_v2.json")

# Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train Logistic Regression Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
with open("./models/random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("./models/random_forest_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Evaluate Model
if __name__ == "__main__":
    evaluate_model_validation(model, vectorizer, X_val, y_val)