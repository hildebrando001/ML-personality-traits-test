import pandas as pd
import streamlit as st

from questions_lists import engList

st.write("""
# Five Big Personality Traits

""")

st.sidebar.header('Answer the questions below')

def user_input_features(qlist):
    questionsDict = {}
    for q in qlist:
        questionsDict[q] = st.sidebar.slider(q, 1, 5, 3)
    questionsDF = pd.DataFrame(questionsDict, index=[0])
    return questionsDF


df = user_input_features(engList)
st.subheader('User inputs')
st.write(df)
