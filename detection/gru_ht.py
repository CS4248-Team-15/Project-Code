import optuna
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from data_utils import load_and_prepare_data

# Load and prepare data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data("./data/Sarcasm_Headlines_Dataset_v2.json")

# Tokenization settings
max_words = 15000  
tokenizer = pickle.load(open("./models/gru_tokenizer.pkl", "rb"))  # Load pre-saved tokenizer
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

# Padding settings
max_len = 50
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')


# Define the objective function that Optuna will optimize
def objective(trial):
    # Hyperparameters to tune
    gru_units = trial.suggest_int('gru_units', 64, 256, step=64)
    embedding_dim = trial.suggest_int('embedding_dim', 256, 4096, step=256)
    epochs = trial.suggest_int('epochs', 2, 5)
    batch_size = trial.suggest_int('batch_size', 32, 64, step=32)

    # Build the Keras model
    model = Sequential([
        Embedding(input_dim=15000, output_dim=embedding_dim, input_length=50),
        SpatialDropout1D(0.2),
        Bidirectional(GRU(units=gru_units, dropout=0.2, recurrent_dropout=0.2)),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Fit the model
    history = model.fit(X_train_pad, np.array(y_train), epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_val_pad, np.array(y_val)), verbose=0)

    # Return the validation accuracy for Optuna to optimize
    val_accuracy = history.history['val_accuracy'][-1]

    print(f"Trial {trial.number}: Validation accuracy={val_accuracy}")
    
    # Clear session to free up memory
    K.clear_session()
    
    return val_accuracy

# Create an Optuna study
optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(direction='maximize')  # Maximize validation accuracy
study.optimize(objective, n_trials=10)  # Number of trials

# Print the best hyperparameters and best score found
print("Best Hyperparameters:", study.best_params)
print("Best Validation Accuracy:", study.best_value)
