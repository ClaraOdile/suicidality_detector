import numpy as np
import pandas as pd
import os

from pathlib import Path
from colorama import Fore, Style

import tensorflow as tf

import sys
sys.path.append(Path(os.getcwd()).parent.absolute())

from params import *

from ml_logic.registry import save_model, load_model
from ml_logic.data import preprocessing
from ml_logic.model import initialize_model, compile_model, train_model

from transformers import TFDistilBertForSequenceClassification

def preprocess(csv_file):
    csv_path = os.path.join(CSV_DIR, csv_file)
    df = pd.read_csv(csv_path)

    data_texts = df['Post'].to_list()
    data_labels = df['Label'].astype('category').cat.codes
    labels = tf.keras.utils.to_categorical(data_labels, num_classes=5)

    encoded = preprocessing(data_texts, stopword=False)

    return encoded, labels

def train(encoded, labels):
    model = load_model()
    if model is None:
        model = initialize_model(model_name = 'distilbert-base-uncased')

    model = compile_model(model, learning_rate=0.005)
    model, history = train_model(model,
            encoded,
            labels
            )

    print("✅ train() done \n")
    save_model(model)

    return model, history

def pred(X:list) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """

    print("\n⭐️ Use case: predict")

    if X is None:
        print('No Post found')
        return None

    model = TFDistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    assert model is not None

    encoded_X = preprocessing(X)
    y_pred = model.predict(encoded_X['input_ids'])

    print("\n✅ prediction done: ", y_pred ,"\n")
    return y_pred['logits']


if __name__ == '__main__':
    post_pred = pred(['you sld not try to kill yourself'])
    print(post_pred)
    #encoded, labels = preprocess('suicidality_dataset.csv')
    #model, history = train(encoded, labels)
