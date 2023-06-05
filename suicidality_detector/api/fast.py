import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Suicidality_Detector.ml_logic.preprocessor import preprocess
from Suicidality_Detector.ml_logic.registry import load_model

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.model = load_model()

@app.get("/predict")
def predict(
        user_text: str,  # ex) I wanna kill myself :(
    ):      # 1
    
    X_pred = pd.DataFrame(locals(), index=[0])

    model = app.state.model
    assert model is not None

    X_processed = preprocess(X_pred)

    y_pred = model.predict(X_processed)

    # y_pred contains dictionary of scores for: 'Supportive', 'Ideation', 'Behavior', 'Attempt', 'Indicator'

    max_val = max(y_pred)
    max_val_p = y_pred[max_val]

    if max_val == 'Supportive':
        return dict('category' = 'Supportive', 'accuracy' = max_val_p, 'explanation' = 'description on Supportive')
    
    elif max_val == 'Ideation':
        return dict('max_val' = 'Ideation', 'accuracy' = max_val_p, 'explanation' = 'description on Ideation')
    
    elif max_val == 'Behavior':
        return dict('max_val' = 'Behavior', 'accuracy' = max_val_p, 'explanation' = 'description on Behavior')
    
    elif max_val == 'Attempt':
        return dict('max_val' = 'Attempt', 'accuracy' = max_val_p, 'explanation' = 'description on Attempt')
    
    elif max_val == 'Indicator':
        return dict('max_val' = 'Indicator', 'accuracy' = max_val_p, 'explanation' = 'description on Indicator')
    
    else:
        return dict('max_val' = 'error', 'accuracy' = 'error', 'explanation' = 'error')


@app.get("/")
def root():
    return {'greeting': 'Hello'}
