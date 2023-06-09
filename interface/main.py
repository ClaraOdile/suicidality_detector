import numpy as np
import pandas as pd
import os

from pathlib import Path
from colorama import Fore, Style

import tensorflow as tf

import sys
sys.path.append(Path(os.getcwd()).parent.absolute())

from params import *
from sklearn.model_selection import train_test_split

from ml_logic.registry import save_model, load_model
from ml_logic.data import clean_data, preprocessing
from ml_logic.model import initialize_model, compile_model, train_model, evaluate_model

def preprocess(csv_file):
    csv_path = os.path.join(CSV_DIR, csv_file)
    df = pd.read_csv(csv_path)

    data_texts = df['Post'].to_list()
    data_labels = df['Label'].astype('category').cat.codes
    labels = tf.keras.utils.to_categorical(data_labels, num_classes=5)

    # train_texts, val_texts, train_labels, val_labels = train_test_split(data_texts, data_labels, test_size=0.1, random_state=42)
    # train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.01, random_state=42)

    train_text_clean = clean_data(data_texts)
    #val_text_clean = clean_data(data_labels)

    # save cleaned X data : ideally concat with Label and then save
    # result_csv_path = os.path.join(CSV_DIR, 'clean.csv')
    # X_clean.to_csv(result_csv_path)
    encodings = preprocessing(train_text_clean)
    #val_encodings = preprocessing(val_text_clean)

    # save clean csv file to CSV_DIR folder
    # csv_save = os.path.join(CSV_DIR, 'result.csv')
    # prep_data.to_csv(csv_save)

    return encodings, labels

def train(encodings, labels):
    # model = load_model()
    # if model is None:
    model = initialize_model(model_name = 'roberta-base')

    model = compile_model(model, learning_rate=0.005)
    model, history = train_model(model,
            encodings,
            labels,
            epochs=100,
            batch_size=16,
            )

    save_model(model)
    # val_mae = np.min(history.history['val_mae'])

    print("✅ train() done \n")

    # return val_mae

def pred(post: str) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if post is None:
        post='This is test message'

    model = load_model()
    assert model is not None

    encoded_inputs = preprocessing(clean_data([post]))
    y_pred = model.predict(encoded_inputs)

    print("\n✅ prediction done: ", y_pred, y_pred.shape, "\n")
    return y_pred


if __name__ == '__main__':
    encodings, labels = preprocess('suicidality_dataset.csv')
    train(encodings, labels)
    pred()
