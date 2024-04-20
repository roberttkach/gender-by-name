import numpy as np
import tensorflow as tf
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def train_model(names, genders_bin, longest_word):
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(names)
    sequences = tokenizer.texts_to_sequences(names)

    X = pad_sequences(sequences, maxlen=len(longest_word))

    X_train, X_test, y_train, y_test = train_test_split(
        X, genders_bin, test_size=0.2, random_state=42
    )

    model = Sequential([
        Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=len(longest_word)),
        LSTM(64, return_sequences=True),
        Dropout(0.5),
        LSTM(32),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
        metrics=['accuracy']
    )

    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    model.fit(
        X_train, np.array(y_train),
        validation_data=(X_test, np.array(y_test)),
        epochs=12, callbacks=[early_stopping]
    )

    model.save('models/gender_by_name.h5')

    return tokenizer
