import streamlit as st
import pandas as pd
import plotly.express as px
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
            return {'category' : 'Supportive', 'probability' : max_val_p, 'explanation' : 'description on Supportive'}
        
        elif max_val == 2:
            return {'max_val' : 'Ideation', 'probability' : max_val_p, 'explanation' : 'description on Ideation'}
        
        elif max_val == 1:
            return {'max_val' : 'Behavior', 'probability' : max_val_p, 'explanation' : 'description on Behavior'}
        
        elif max_val == 0:
            return {'max_val' : 'Attempt', 'probability' : max_val_p, 'explanation' : 'description on Attempt'}
        
        elif max_val == 3:
            return {'max_val' : 'Indicator', 'probability' : max_val_p, 'explanation' : 'description on Indicator'}
        else:
            return {'max_val' : 'error', 'probability' : 'error', 'explanation' : 'error'}

    prediction = classifier(max_val)
    print(prediction)

    # result as the output class
    if max_val < 4:
        st.markdown(f"<h2 style='text-align: center; color: red;'>Suicidal {prediction['max_val']}</h2>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h2 style='text-align: center; color: red;'>{prediction['max_val']}</h2>", unsafe_allow_html=True)

    # short interpretation of the prediction
    st.markdown(f"## Interpretation: ")
    st.markdown(f"<div style='font-size: 16px;'>{prediction['explanation']}</div>", unsafe_allow_html=True)

    # Show the scores for each class as a fancy bar chart
    graph_df = pd.DataFrame.from_dict(prediction, orient='index').reset_index()
    graph_df.columns=['Category', 'probability']
    fig = px.bar(graph_df, y='Category', x='probability', color='Category', orientation='v',
                 text='probability', title='Category probability', color_discrete_sequence=color_sequence)
    fig.update_layout(xaxis_title="probability", yaxis_title="Category", template='plotly_white', autosize=True)
    st.plotly_chart(fig)

    # Display the recommendation area, with placeholder text for now
    st.subheader("Recommendation:")
    st.markdown("<div style='font-size: 16px;'>Clara's recommendation text ... .</div>", unsafe_allow_html=True)

# Display the disclaimer
st.markdown("<div style='font-size: 14px; color: gray;'>DISCLAIMER: Making wrong predictions could lead to severe consequences. This application is no replacement for professionals, but rather supports psychiatric care.</div>", unsafe_allow_html=True)
