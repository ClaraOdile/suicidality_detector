import numpy as np
import time

from colorama import Fore, Style
from typing import Tuple
from params import *

# Timing the TF import
print(Fore.BLUE + "\nLoading TensorFlow..." + Style.RESET_ALL)
start = time.perf_counter()

import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, TFTrainer, TFTrainingArguments

from keras import Model
from keras.callbacks import EarlyStopping

end = time.perf_counter()
print(f"\n✅ TensorFlow loaded ({round(end - start, 2)}s)")

def initialize_model(model_name = 'roberta-base') -> Model:
    """
    Initialize the Neural Network with random weights
    """
    model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)
    print("✅ Model initialized")

    return model

def compile_model(model: Model, learning_rate=0.005) -> Model:
    """
    Compile the Neural Network
    """
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    loss = model.compute_loss
    model.compile(optimizer=optimizer,
            loss=loss,
            metrics=['accuracy'])

    print("✅ Model compiled")

    return model

def train_model(
        model: Model,
        train_dataset,
        val_dataset,
        epochs=100,
        batch_size=16,
        patience=10
    ) -> Tuple[Model, dict]:
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    print(Fore.BLUE + "\nTraining model..." + Style.RESET_ALL)

    es = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    training_args = TFTrainingArguments(
        output_dir=MODEL_DIR,          # output directory
        num_train_epochs=epochs,              # total number of training epochs
        per_device_train_batch_size=batch_size,  # batch size per device during training
        per_device_eval_batch_size=64,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir=os.path.join(MODEL_DIR, 'logs')         # directory for storing logs
    )

    with training_args.strategy.scope():
        trainer_model = model

    trainer = TFTrainer(
        model=trainer_model,                 # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=val_dataset             # evaluation dataset
    )

    trainer.train()
    trainer.evaluate()

    print(f"✅ Model trained on {len(X)} rows with min val MAE: {round(np.min(history.history['val_mae']), 2)}")

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
