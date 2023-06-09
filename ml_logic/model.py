import numpy as np
import pandas as pd
import time

from colorama import Fore, Style
from typing import Tuple
from params import *

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

import tensorflow as tf
from transformers import DistilBertTokenizer,TFDistilBertForSequenceClassification

from keras import Model
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")

def initialize_model(model_name = 'distilbert-base-uncased') -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)
    print("✅ Model initialized")

    return model

def compile_model(model: Model, learning_rate=0.005) -> Model:
    """
    Compile the Neural Network
    """
    # optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    # loss = model.hf_compute_loss
    #f1_score = tfa.metrics.F1Score(num_classes=5, average='macro')
    # precision = tf.keras.metrics.Precix
    recall = tf.keras.metrics.Recall()
    # accuracy = tf.keras.metrics.Accuracy()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[recall])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        encodings,
        labels
    ):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    train_input_ids = encodings['input_ids']
    train_attention_mask = encodings['attention_mask']

    # train_dataset = tf.data.Dataset.from_tensor_slices((
    #     dict(train_encodings),
    #     train_labels
    # ))
    # val_dataset = tf.data.Dataset.from_tensor_slices((
    #     dict(val_encodings),
    #     val_labels
    # ))

    # es = EarlyStopping(
    #     monitor="val_loss",
    #     patience=patience,
    #     restore_best_weights=True,
    #     verbose=1
    # )

    history = model.fit({'input_ids': train_input_ids,
                         'attention_mask': train_attention_mask},
                        labels,
                        epochs=5,
                        batch_size=128,
                        validation_split=0.2
                    )

    #print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
    ) -> Tuple[Model, dict]:
    """
    Evaluate trained model performance on the dataset
    """

    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print(f"\n❌ No model to evaluate")
        return None

    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        # callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
