import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import pickle
from data_utils import load_and_prepare_data

def evaluate_model_test(model, X_test, y_test):
    with open("./models/lstm_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    y_test_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
    print("LSTM Test Set Performance:")
    print(classification_report(y_test, y_test_pred))

# Load and prepare data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data("./data/Sarcasm_Headlines_Dataset_v2.json")

# Tokenize text
max_words = 15000  # Max number of words to keep in the vocabulary
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert texts to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

# Pad sequences to ensure uniform length
max_len = 50  # Set a reasonable sequence length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')

# Save tokenizer for future use
with open('./models/lstm_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Create LSTM model
model = Sequential([
    Embedding(input_dim=max_words, output_dim=128),
    SpatialDropout1D(0.2),  # Helps prevent overfitting
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(1, activation='sigmoid')  # Binary classification (sigmoid for probabilities)
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.summary()

if __name__ == "__main__":
    print("running")

    # Train the model
    history = model.fit(
        X_train_pad, np.array(y_train),
        validation_data=(X_val_pad, np.array(y_val)),
        epochs=3,
        batch_size=32
    )

    model.save("./models/lstm_model.keras")