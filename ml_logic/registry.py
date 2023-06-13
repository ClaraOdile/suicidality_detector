import os
import time
from pathlib import Path
#import pickle

from colorama import Fore, Style
from tensorflow import keras
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification
#from google.cloud import storage

from params import *

# def save_model(model: keras.Model = None) -> None:
#     """
#     Persist trained model locally on the hard drive at f"{MODEL_DIR/{timestamp}.h5"
#     """

#     timestamp = time.strftime("%Y%m%d-%H%M%S")

#     # Save model locally
#     model_path = os.path.join(MODEL_DIR, f"{timestamp}.h5")
#     model.save(model_path)

#     print("✅ Model saved locally")

#     return None


def load_model():
    """
    Return a saved model:
    - locally (latest one in alphabetical order)

    Return None (but do not Raise) if no model is found

    """
    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    MODEL_DIR = os.path.join(Path(os.getcwd()).parent.absolute(), 'models')

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    model = TFRobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=5)
    local_model_paths = os.path.join(MODEL_DIR, 'weights', 'weights')
    model.load_weights(local_model_paths)


    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    print("✅ Model loaded from local disk")

    return tokenizer, model
