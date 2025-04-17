import logistic_regression
import random_forest
import lstm
import lstm2
import gru
import gru2
from data_utils import load_and_prepare_data
import pickle
from tensorflow.keras.models import load_model

def main():
    _, _, X_test, _, _, y_test = load_and_prepare_data("./data/Sarcasm_Headlines_Dataset_v2.json")

    # List of models and their paths to be evaluated
    model_files = [
        ("logistic_regression", "logistic_regression_model.pkl"),
        ("random_forest", "random_forest_model.pkl"),
        ("lstm", "lstm_model.pkl"),
        ("lstm2", "lstm2_model.pkl")
        ("gru", "gru_model.pkl"),
        ("gru2", "gru2_model.pkl")
    ]

    # Append the models file folder to each model address
    for model_name, model_file in model_files:
        model_file = "./models/" + model_file

        if model_name == "logistic_regression":
            with open("./models/logistic_regression_model.pkl", "rb") as f:
                model = pickle.load(f)
            logistic_regression.evaluate_model_test(model, X_test, y_test)
        elif model_name == "random_forest":
            with open("./models/random_forest_model.pkl", "rb") as f:
                model = pickle.load(f)
            random_forest.evaluate_model_test(model, X_test, y_test)
        elif model_name == "lstm":
            model = load_model("./models/lstm_model.keras")
            lstm.evaluate_model_test(model, X_test, y_test)
        elif model_name == "lstm2":
            model = load_model("./models/lstm2_model.keras")
            lstm2.evaluate_model_test(model, X_test, y_test)
        elif model_name == "gru":
            model = load_model("./models/gru_model.keras")
            gru.evaluate_model_test(model, X_test, y_test)
        elif model_name == "gru2":
            model = load_model("./models/gru2_model.keras")
            gru2.evaluate_model_test(model, X_test, y_test)

if __name__ == "__main__":
    main()