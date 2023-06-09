import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random

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
    # input = clean_data(user_text)
    # encoded = preprocessing(input)

    # model = load_model()

    # predicted = model.predict([encoded['input_ids'], encoded['attention_mask']])
    # print(predicted)

    #dummy data
    ## dummy input for test ##
    probabilities = [0.0, 0.0, 0.0, 0.0, 0.0]
    probabilities = [round(random.uniform(0.01, 0.99), 2) for i in range(0,5)]
    labels = [0, 1, 2, 3, 4]
    response = {label: prob for label, prob in zip(labels, probabilities)}
    ## dummy input for test ##

    return response

    ## 'response' contains dictionary of scores for: 'Supportive(4)', 'Ideation(2)', 'Behavior(1)', 'Attempt(0)', 'Indicator(3)'



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
#uvicorn main:app --reload
