import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
# from transformers import TFRobertaModel, RobertaTokenizer
import tensorflow as tf
import tensorflow_addons as tfa
import ml_logic

# from Suicidality_Detector.ml_logic.model import load_model

app = FastAPI()

@app.get('/')
def index():
    return {'message': 'Hello, World'}

@app.get('/Welcome')
def get_name(name: str):
    return {'Welcome To Suicidality Detector': f'{name}'}

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(
        user_text: str,  # ex) I wanna kill myself :(
    ):
    
    # inputs = pd.DataFrame(locals(), index=[0])

    # model = ml_logic.model.Distilbert
    # model = ml_logic.model.Roberta
    # tokenizer = ml_logic.Preprocess.tokenizer
    # assert model is not None

    # # Tokenize the input sentence
    # inputs = tokenizer(inputs, truncation=True, padding=True, return_tensors="tf")

    # # Make the prediction
    # logits = model(inputs.input_ids, attention_mask=inputs.attention_mask).logits
    # probabilities = tf.nn.softmax(logits, axis=1)[0].numpy().tolist()


    #dummy data
    probabilities = [0.9, 0.8, 0.99, 0.5, 0.1]

    # Define class labels
    labels = [0, 1, 2, 3, 4]

    # Prepare the API response
    response = {label: prob for label, prob in zip(labels, probabilities)}

    return response

    ## 'response' contains dictionary of scores for: 'Supportive(4)', 'Ideation(2)', 'Behavior(1)', 'Attempt(0)', 'Indicator(3)'



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn main:app --reload
