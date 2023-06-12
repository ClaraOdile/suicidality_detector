import streamlit as st
from fast import predict
import requests
import json

# color sequence
color_sequence = ['#AFEEEE', '#48D1CC', '#40E0D0', '#00CED1', '#20B2AA']
# text input cell for users
user_post = st.text_input("Enter Text:")

# button uses the fast_predict function to get a prediction
if st.button("Predict"):
    st.markdown(user_post)
    url = 'http://127.0.0.1:8000/predict'  # uvicorn web server url
    params= {'post': user_post}
    response = requests.get(url, params=params)
    results = response.json()[0]

    max_val = int(results['max_val'])
    max_val_p = float(results['max_val_p'])

    def classifier(max_val):
        if max_val == 4:
            return {'max_val' : 'Supportive', 'probability' : max_val_p, 'explanation' : 'The author of this text shows empathy toward another person regarding suicidality and offers help or advice'}

        elif max_val == 2:
            return {'max_val' : 'Ideation', 'probability' : max_val_p, 'explanation' : 'This text contains thoughts about suicide, ranging from passing considerations to detailed plans'}

        elif max_val == 1:
            return {'max_val' : 'Behavior', 'probability' : max_val_p, 'explanation' : 'This text reports actions that are related to suicide but do not constitute an actual attempt, such as preparing to commit suicide.'}

        elif max_val == 0:
            return {'max_val' : 'Attempt', 'probability' : max_val_p, 'explanation' : 'This text reports a suicide attempt, that is, a specific act with the intention of dying in the process.'}

        elif max_val == 3:
            return {'max_val' : 'Indicator', 'probability' : max_val_p, 'explanation' : 'This text indicates a potential suicide risk without specifically mentioning suicide.'}
        else:
            return {'max_val' : 'error', 'probability' : 'error', 'explanation' : 'error'}

    prediction = classifier(max_val)
    st.markdown(f"This assessment is based on an AI analysis and cannot replace a psychiatric/psychological evaluation!")

    st.markdown(f"<h2 style='text-align: center; color: red;'>{prediction['max_val']}</h2>", unsafe_allow_html=True)

    st.markdown(f"## Interpretation: ")
    st.markdown(f"<div style='font-size: 16px;'>{prediction['explanation']}</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='font-size: 14px; color: gray;'>DISCLAIMER: Making wrong predictions could lead to severe consequences. This application is no replacement for professionals, but rather supports psychiatric care.</div>", unsafe_allow_html=True)
