import streamlit as st
from fast import predict

# color sequence
color_sequence = ['#AFEEEE', '#48D1CC', '#40E0D0', '#00CED1', '#20B2AA']
# text input cell for users
user_input = st.text_input("Enter Text:")

# button uses the fast_predict function to get a prediction
if st.button("Predict"):
    response = predict(user_input)

    max_val_p = max(response.values())

    for labels, probabilities in response.items():
        if probabilities == max_val_p:
            max_val = labels

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
