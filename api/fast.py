import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import tensorflow_addons as tfa
from ml_logic.registry import load_model
from params import *
from transformers import TFRobertaModel, RobertaTokenizer
import json
from pathlib import Path
import os

app = FastAPI()

tokenizer, model = load_model()
print('model is loaded')

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

# http://127.0.0.1:8000/predict?post=string
@app.get('/predict')
def predict(post):
    print('Prediction API Call started...')

    inputs = preprocess(post)
    input_ids = input["input_ids"]
    attention_mask = input["attention_mask"]

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

    ## 'response' contains dictionary of scores for: 'Attempt(0)', 'Behavior(1)', 'Ideation(2)', 'Indicator(3)', 'Supportive(4)'
