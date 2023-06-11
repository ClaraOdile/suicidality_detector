import numpy as np
from colorama import Fore, Style
from params import *

import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification
from keras import Model
from keras.callbacks import EarlyStopping

def initialize_model(model_name):
    """
    Initialize the pre-trained model with random weights
    """
    model = TFDistilBertForSequenceClassification.from_pretrained(model_name, num_labels=5)
    print("✅ Model initialized")

    return model

def compile_model(model: Model, learning_rate=0.005):
    """
    Compile the pre-trained model
    """
    precision = tf.keras.metrics.Precision()
    recall = tf.keras.metrics.Recall()

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=[recall, precision])
    print("✅ Model compiled")

    return model

def train_model(
        model,
        encoded,
        labels
    ):
    """
    Fit the model and return fitted_model and history
    """
    print('✅ Training model...')
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    es = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit({'input_ids': input_ids, 'attention_mask': attention_mask},
                    labels,
                    epochs=100,
                    batch_size=16,
                    validation_split=0.2,
                    callbacks=[es]
                )
    #print(f"✅ Model trained on {len(input_ids)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

    return model, history
