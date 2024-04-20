import tkinter as tk
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


def predict_gender(model, tokenizer, longest_word):
    test_name = name_entry.get()
    sequence = tokenizer.texts_to_sequences([test_name])
    X_custom = pad_sequences(sequence, maxlen=len(longest_word))
    prediction = model.predict(X_custom)

    if prediction[0][0] > 0.5:
        result_label.config(text=f"{test_name} - M     {prediction[0][0]}")
    else:
        result_label.config(text=f"{test_name} - F     {prediction[0][0]}")


def run_prediction(tokenizer, longest_word):
    model = tf.keras.models.load_model('gender_by_name.h5')
    window = tk.Tk()

    global name_entry
    name_entry = tk.Entry(window)
    name_entry.pack()

    predict_button = tk.Button(
        window, text="Predict Gender",
        command=lambda: predict_gender(model, tokenizer, longest_word)
    )
    predict_button.pack()

    global result_label
    result_label = tk.Label(window, text="")
    result_label.pack()

    window.mainloop()
