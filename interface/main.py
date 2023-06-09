import numpy as np
import pandas as pd
import os

from pathlib import Path
from colorama import Fore, Style

import tensorflow as tf

# set parent folder(suicidality_detector) to sys.path to use functions in another subdirectories
import sys
sys.path.append(Path(os.getcwd()).parent.absolute())

from params import *

from ml_logic.registry import save_model, load_model
from ml_logic.data import clean_data, preprocessing
from ml_logic.model import initialize_model, compile_model, train_model, evaluate_model

def preprocess(csv_file):
    csv_path = os.path.join(CSV_DIR, csv_file)
    df = pd.read_csv(csv_path)

    data_texts = df['Post'].to_list()
    data_labels = df['Label'].astype('category').cat.codes
    labels = tf.keras.utils.to_categorical(data_labels, num_classes=5)

    train_text_clean = clean_data(data_texts)
    encodings = preprocessing(train_text_clean)

    return encodings, labels

def train(encodings, labels):
    # model = load_model()
    # if model is None:
    model = initialize_model(model_name = 'roberta-base')

    model = compile_model(model, learning_rate=0.005)
    model, history = train_model(model,
            encodings,
            labels,
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
