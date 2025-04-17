import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from data_utils import load_and_prepare_data

def evaluate_model_validation(model, vectorizer, X_val, y_val):
    """Evaluates a trained model on validation and test sets."""
    X_val_tfidf = vectorizer.transform(X_val)

    y_val_pred = model.predict(X_val_tfidf)

    print("Logistic Regression Validation Set Performance:")
    print(classification_report(y_val, y_val_pred))

def evaluate_model_test(model, X_test, y_test):
    with open("./models/logistic_regression_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
        X_test_tfidf = vectorizer.transform(X_test)
    y_test_pred = model.predict(X_test_tfidf)
    print("Logistic Regression Test Set Performance:")
    print(classification_report(y_test, y_test_pred))


# Load and prepare data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data("./data/Sarcasm_Headlines_Dataset_v2.json")

# Feature Extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
with open("./models/logistic_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("./models/logistic_regression_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Evaluate Model
if __name__ == "__main__":
    evaluate_model_validation(model, vectorizer, X_val, y_val)