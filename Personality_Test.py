import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans

from questions_lists import engList

st.write("""
# Five Big Personality Traits

""")

def get_data(file):
    data = pd.read_csv(file)
    return data


st.sidebar.header('Answer the questions')

def user_input_features(qlist):
    questionsDict = {}
    for q in qlist:
        questionsDict[q] = st.sidebar.slider(q, 1, 5, 5)
    questionsDF = pd.DataFrame(questionsDict, index=[0])
    return questionsDF


data = get_data("data-final.csv")


df_questions = user_input_features(engList)
# st.subheader('User inputs')

def run_algorithm():
    kmeans = KMeans(n_clusters=5)
    k_fit = kmeans.fit(data)
    predict = k_fit.predict(df_questions)

    st.write(predict)

if st.sidebar.button('Analyse'):
    run_algorithm()