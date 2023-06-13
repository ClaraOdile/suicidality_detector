import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from ml_logic.data import preprocessing
from params import *
from pydantic import BaseModel
from transformers import TFDistilBertForSequenceClassification, TFRobertaForSequenceClassification
import json
from pathlib import Path
import os

class Item(BaseModel):
    category: str
    proba: float

app = FastAPI()

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

# http://127.0.0.1:8000/predict?post=string
@app.get('/predict')
def predict(post):
    print('Prediction API Call started...')

    # this pred() function needs to be fixed

    loaded_model = TFRobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=5)
    model_dir= os.path.join(Path(os.getcwd()).parent.absolute(), 'models')
    loaded_model.load_weights(os.path.join(model_dir, 'weights', 'weights'))

    encoded = preprocessing([post])
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    predicted = loaded_model.predict([input_ids, attention_mask])

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
