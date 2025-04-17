import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (TextVectorization, Embedding, SpatialDropout1D, Bidirectional, GRU, 
                                     Conv1D, GlobalMaxPooling1D, Dense, Dropout, 
                                     BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import pickle
from data_utils import load_and_prepare_data

# If loading pre-trained model from another file can import and use this function
def evaluate_model_test(model, X_test, y_test):
    y_test_pred = (model.predict(np.array(X_test)) > 0.5).astype("int32")
    print("GRU Test Set Performance:")
    print(classification_report(y_test, y_test_pred))
    # missing_labels = np.setdiff1d(y_test, y_test_pred)
    # print("Labels not predicted:", missing_labels)

X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data("./data/Sarcasm_Headlines_Dataset_v2.json")

max_words = 15000  # Max number of words to keep in the vocabulary
max_len = 50 
vectorizer = TextVectorization(
    max_tokens=max_words,
    output_mode='int',
    output_sequence_length=max_len,
)

vectorizer.adapt(X_train)

model = Sequential([
    vectorizer,
    Embedding(input_dim=15000, output_dim=64, input_length=50),
    SpatialDropout1D(0.2),

    Conv1D(64, kernel_size=3, activation='relu', padding='same'), 
    # Conv1D(64, kernel_size=3, activation='relu', padding='same'), 

    Bidirectional(GRU(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    # Bidirectional(GRU(24, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    GlobalMaxPooling1D(),

    Dense(128, activation='relu'),
    Dropout(0.3),
    BatchNormalization(),

    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()

if __name__ == "__main__":
    print("running")

    callback = EarlyStopping(monitor='loss')

    history = model.fit(
        np.array(X_train), np.array(y_train),
        validation_data=(np.array(X_val), np.array(y_val)),
        epochs=2,
        batch_size=32,
        callbacks=[callback]
    )

    model.save("./models/gru2_model.keras")

    y_test_pred = (model.predict(np.array(X_test)) > 0.5).astype("int32")
    print("GRU Test Set Performance:")
    print(classification_report(y_test, y_test_pred))

    # Best Hyperparameters: {'embedding_dim': 64, 'conv_units': 32, 'gru_units': 32, 'gru_units2': 24, 'relu_units': 352, 'epochs': 4, 'batch_size': 32}