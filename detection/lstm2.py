import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (TextVectorization, Embedding, Conv1D, Bidirectional, LSTM, Dense, Dropout, 
                                     SpatialDropout1D, GlobalMaxPooling1D, Attention, BatchNormalization, Layer, Multiply)
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import classification_report
from data_utils import load_and_prepare_data

def evaluate_model_test(model, X_test, y_test):
    y_test_pred = (model.predict(np.array(X_test)) > 0.5).astype("int32")
    print("LSTM2 Test Set Performance:")
    print(classification_report(y_test, y_test_pred))
    # missing_labels = np.setdiff1d(y_test, y_test_pred)
    # print("Labels not predicted:", missing_labels)

# Load and prepare data
X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data("./data/Sarcasm_Headlines_Dataset_v2.json")

# Tokenize text
max_words = 15000  # Max number of words to keep in the vocabulary
max_len = 50  # Set a reasonable sequence length
vectorizer = TextVectorization(
    max_tokens=max_words,        # Number of words in the vocabulary
    output_mode='int',           # Output integers (instead of embeddings)
    output_sequence_length=max_len,  # Pad or truncate sequences to this length
    standardize='lower_and_strip_punctuation'
)

vectorizer.adapt(X_train)

# class BahdanauAttention(Layer):
#     def __init__(self, units):
#         super(BahdanauAttention, self).__init__()
#         self.W1 = tf.keras.layers.Dense(units)
#         self.W2 = tf.keras.layers.Dense(units)
#         self.V = tf.keras.layers.Dense(1)

#     def call(self, values):
#         # Add time axis to hidden state
#         hidden_with_time_axis = tf.expand_dims(values[:, -1, :], axis=1)
        
#         # Calculate attention scores
#         score = self.V(tf.nn.tanh(
#             self.W1(values) + self.W2(hidden_with_time_axis)))
        
#         # Get attention weights
#         attention_weights = tf.nn.softmax(score, axis=1)
        
#         # Calculate context vector
#         context_vector = attention_weights * values
#         context_vector = tf.reduce_sum(context_vector, axis=1)
#         return context_vector

from tensorflow.keras.layers import Layer, Multiply
import tensorflow.keras.backend as K

class ContextualAttention(Layer):
    def __init__(self, **kwargs):
        super(ContextualAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        # Learnable weights
        self.W = self.add_weight(
            name="att_weight", 
            shape=(input_shape[-1], 1),  # (hidden_dim, 1)
            initializer="glorot_uniform"
        )
        self.b = self.add_weight(
            name="att_bias", 
            shape=(input_shape[1], 1),  # (seq_len, 1)
            initializer="zeros"
        )
        super(ContextualAttention, self).build(input_shape)

    def call(self, x):
        # Attention scores (batch_size, seq_len, 1)
        e = K.tanh(K.dot(x, self.W) + self.b)
        
        # Attention weights (softmax over sequence axis)
        a = K.softmax(e, axis=1)
        
        # Apply attention weights
        output = Multiply()([x, a])
        
        # Sum over sequence (batch_size, hidden_dim)
        return K.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])  # Removes seq_len dimension

model = Sequential([
    vectorizer,
    Embedding(max_words, 64, input_length=max_len),
    SpatialDropout1D(0.2),
    
    Conv1D(64, 3, activation='relu', padding='same'),
    Conv1D(64, 5, activation='relu', padding='same'),
    Conv1D(64, 7, activation='relu', padding='same'),
    
    Bidirectional(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
    
    ContextualAttention(),  # Custom attention layer
    Dense(128, activation='relu'),
    Dropout(0.2),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

# optimizer = AdamW(learning_rate=3e-5, weight_decay=1e-4)
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.summary()

if __name__ == "__main__":
    print("running")

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    callbacks = [lr_scheduler]

    # Train the model
    history = model.fit(
        np.array(X_train), np.array(y_train),
        validation_data=(np.array(X_val), np.array(y_val)),
        epochs=5,
        batch_size=32,
        callbacks=callbacks
    )

    model.save("./models/lstm2_model.keras")

    y_test_pred = (model.predict(np.array(X_test)) > 0.5).astype("int32")
    print("LSTM2 Test Set Performance:")
    print(classification_report(y_test, y_test_pred))


