import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import TFRobertaModel, RobertaTokenizer
import tensorflow as tf
import tensorflow_addons as tfa
from ml_logic.registry import load_model

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

@app.get("/preprocess")
def preprocess(user_text):
    inputs = str(user_text)

    tokenizer, model = load_model()
    print('model is loaded')
    assert tokenizer is not None

    # Tokenize the input sentence
    inputs = tokenizer(inputs, truncation=True, padding=True, return_tensors="tf")
    return inputs


@app.get("/predict")
def predict(
        user_text :str,  # ex) I wanna kill myself :(
    ):

    tokenizer, model = load_model()

    inputs = preprocess(user_text)

    # Make the prediction
    logits = model(inputs.input_ids, attention_mask=inputs.attention_mask).logits
    probabilities = tf.nn.softmax(logits, axis=1)[0].numpy().tolist()

    # inputs_ids = inputs["input_ids"]
    # print(inputs_ids)
    # inputs_attention_mask = inputs["attention_mask"]
    # print(inputs_attention_mask)
    # predictions = model.predict([inputs_ids, inputs_attention_mask])
    # print(predictions)

    # Define class labels
    labels = [0, 1, 2, 3, 4]

    # Prepare the API response
    response = {label: prob for label, prob in zip(labels, probabilities)}

    return response

    ## 'response' contains dictionary of scores for: 'Supportive(4)', 'Ideation(2)', 'Behavior(1)', 'Attempt(0)', 'Indicator(3)'



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn main:app --reload
