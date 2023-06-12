import pandas as pd
import numpy as np
from params import *
from pydantic import BaseModel
import json
from pathlib import Path
import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
# from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification
from transformers import TFRobertaModel, RobertaTokenizer
import tensorflow as tf
import tensorflow_addons as tfa
from ml_logic.registry import load_model

# from Suicidality_Detector.ml_logic.model import load_model

class Item(BaseModel):
    category: str
    proba: float

app = FastAPI()

tokenizer, model = load_model()

@app.get('/')
def index():
    return {'message': 'Hello, World'}


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/preprocess")
def preprocess(post):
    inputs = str(post)
    print('model is loaded')
    assert tokenizer is not None
    # Tokenize the input sentence
    inputs = tokenizer(inputs, truncation=True, padding=True, return_tensors="tf")
    return inputs


@app.get("/predict")
def predict(
        post :str,  # ex) I wanna kill myself :(
    ):
    print('Prediction API Call started...')
    print('Encoding Inputs...')
    inputs = preprocess(post)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Make the prediction
    print('Making Prediction...')
    predicted = model.predict([input_ids, attention_mask])
    probabilities = predicted['logits']
    max_val = probabilities.argmax()
    max_val_p = probabilities.max()
    print(f'highest possibility is {max_val_p} on category {max_val}')
    returned = [
        {'max_val' : max_val, 'max_val_p' : max_val_p},
        ]
    json_str = json.dumps(returned, indent=4, default=str)
    return Response(content=json_str, media_type='application/json')

    ## 'response' contains dictionary of scores for: 'Supportive(4)', 'Ideation(2)', 'Behavior(1)', 'Attempt(0)', 'Indicator(3)'



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn main:app --reload
