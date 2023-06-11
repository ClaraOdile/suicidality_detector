import glob
import os
import time
#import pickle

from colorama import Fore, Style
from tensorflow import keras
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

    # Get the latest model version name by the timestamp on disk
    local_model_paths = glob.glob(f"{MODEL_DIR}/*")

    if not local_model_paths:
        return None

    most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    #keras.models.model_from_json(os.path.join(MODEL_DIR, 'config.json'))
    json_file = open(os.path.join(MODEL_DIR, 'config.json'))
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights (os.path.join(MODEL_DIR, "tf_model.h5"))

    #print("Loaded model from disk")
    #latest_model = keras.models.load_model(MODEL_DIR)

    print("✅ Model loaded from local disk")

    return loaded_model
