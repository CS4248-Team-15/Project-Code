import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Embedding, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import pickle
from data_utils import load_and_prepare_data

def evaluate_model_test(model, X_test, y_test):
    with open("./models/gru_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
        X_test_seq = tokenizer.texts_to_sequences(X_test)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post', truncating='post')
    y_test_pred = (model.predict(X_test_pad) > 0.5).astype("int32")
    print(y_test_pred)
    print("GRU Test Set Performance:")
    print(classification_report(y_test, y_test_pred))
    # missing_labels = np.setdiff1d(y_test, y_test_pred)
    # print("Labels not predicted:", missing_labels)

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

model = Sequential([
    Embedding(input_dim=max_words, output_dim=1536),
    SpatialDropout1D(0.2),  # Helps prevent overfitting,
    Bidirectional(GRU(256, dropout=0.2, recurrent_dropout=0.2)),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()

# Save tokenizer for future use
with open('./models/gru_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

if __name__ == "__main__":
    print("running")

    callback = EarlyStopping(monitor='loss')

    # Train the model
    history = model.fit(
        X_train_pad, np.array(y_train),
        validation_data=(X_val_pad, np.array(y_val)),
        epochs=2,
        batch_size=64,
        callbacks=[callback]
    )

    model.save("./models/gru_model.keras")