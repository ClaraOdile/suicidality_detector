import streamlit as st
from fast import predict
import requests
import json
import time


#WALLPAPER
import streamlit as st

# Define the custom styles
custom_css = """
<style>
body {
background-image: url("https://images.pexels.com/photos/1266810/pexels-photo-1266810.jpeg");
background-size: cover;
}
</style>
"""

# Inject the custom CSS
st.markdown(custom_css, unsafe_allow_html=True)

st.title('Suicidality Detector')
#WALLPAPER CODE ENDS


# color sequence
color_sequence = ['#AFEEEE', '#48D1CC', '#40E0D0', '#00CED1', '#20B2AA']
# text input cell for users
user_post = st.text_input("Enter Text:")


def classifier(max_val):
    if max_val == 4:
        return {'max_val' : 'Supportive', 'probability' : max_val_p, 'explanation' : 'The author of this text shows empathy toward another person regarding suicidality and offers help or advice',
                'recommendation' : ''}

    elif max_val == 2:
        return {'max_val' : 'Ideation',
                'probability' : max_val_p,
                'explanation' : 'This text contains thoughts about suicide, ranging from passing considerations to detailed plans',
                'recommendation' : 'There is some risk that the author of this text may commit suicide. If you have the opportunity to contact this person, please offer help and refer to local emergency calls or ambulances.'}

    elif max_val == 1:
        return {'max_val' : 'Behavior',
                'probability' : max_val_p,
                'explanation' : 'This text reports actions that are related to suicide but do not constitute an actual attempt, such as preparing to commit suicide.', 'recommendation' : 'The risk that the author of this text will commit suicide is considered high. If you have the opportunity to contact this person, please offer help and refer to local emergency calls or ambulances.'}

    elif max_val == 0:
        return {'max_val' : 'Attempt',
                'probability' : max_val_p,
                'explanation' : 'This text reports a suicide attempt, that is, a specific act with the intention of dying in the process.',
                'recommendation' : 'The risk that the author of this text will commit suicide is considered high. If you have the opportunity to contact this person, please offer help and refer to local emergency calls or ambulances.'}

    elif max_val == 3:
        return {'max_val' : 'Indicator',
                'probability' : max_val_p,
                'explanation' : 'This text indicates a potential suicide risk without specifically mentioning suicide.',
                'recommendation' : 'There is some risk that the author of this text may commit suicide. If you have the opportunity to contact this person, please offer help and refer to local emergency calls or ambulances.'}
    else:
        return {'max_val' : 'error', 'probability' : 'error', 'explanation' : 'error'}




# button uses the fast_predict function to get a prediction
if st.button("Analyze"):
    #st.markdown(user_post)
    st.info('Please wait...') # This will add a message "Please wait..."
    with st.spinner('Analyzing...'):
        time.sleep(5)


        url = 'http://127.0.0.1:8000/predict'  # uvicorn web server url
        params= {'post': user_post}
        response = requests.get(url, params=params)
        results = response.json()[0]

        max_val = int(results['max_val'])
        max_val_p = float(results['max_val_p'])

        st.success('Analysis complete')
        #st.experimental_rerun() # This will clear the "Please wait..." message.

        prediction = classifier(max_val)

        st.markdown("<br>", unsafe_allow_html=True)  # space
        st.markdown(f"<h2 style='text-align: center; color: red;'>{prediction['max_val']}</h2>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)  # space
        st.markdown(f"<div style='font-size: 16px; color: #FFCCBC; '>{prediction['explanation']}</div>", unsafe_allow_html=True)

        if prediction['recommendation']:
            if prediction['max_val'] in ['Ideation', 'Indicator']:
                st.markdown("<br>", unsafe_allow_html=True)  # space
                st.markdown(f"<div style='font-size: 16px; color: Beige;'>{prediction['recommendation']}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<br>", unsafe_allow_html=True)  # space
                st.markdown(f"<div style='font-size: 16px; color: red;'>{prediction['recommendation']}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)  # space
        st.markdown("<div style='font-size: 16px; color: brown;'><strong>DISCLAIMER:</strong> <em>This assessment is based on an AI analysis and cannot replace a psychiatric/psychological evaluation.</div>", unsafe_allow_html=True)
