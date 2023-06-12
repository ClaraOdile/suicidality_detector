import glob
import os
import time
#import pickle

from colorama import Fore, Style
from tensorflow import keras
from transformers import RobertaTokenizer, TFRobertaForSequenceClassification

#from google.cloud import storage

from params import *

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{MODEL_DIR/{timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(MODEL_DIR, f"{timestamp}.h5")
    model.save(model_path)

    print("✅ Model saved locally")

    return None


def load_model(stage="Production") -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)

    Return None (but do not Raise) if no model is found

    """

    print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    model_name ="roberta-base"

    model = TFRobertaForSequenceClassification.from_pretrained(model_name, num_labels=5)
    model.load_weights('ml_logic/weights/weights')

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    return tokenizer, model
