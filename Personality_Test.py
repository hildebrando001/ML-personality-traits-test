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
        questionsDict[q] = st.sidebar.slider(q, 1, 5, 3)
    questionsDF = pd.DataFrame(questionsDict, index=[0])
    return questionsDF


data = get_data("data-final.csv")


df_questions = user_input_features(engList)


def run_algorithm():
    kmeans = KMeans(n_clusters=5)
    k_fit = kmeans.fit(data)
    predict = k_fit.predict(df_questions)

    # st.subheader('User inputs')
    st.write(df_questions)

    # Selecting columns of each group
    col_list = list(df_questions)
    ext = col_list[:10]
    neu = col_list[10:20]
    agr = col_list[20:30]
    csn = col_list[30:40]
    opn = col_list[40:50]

    data_total = pd.DataFrame()
    data_total['Extroversion'] = df_questions[ext].sum(axis=1)
    data_total['Neuroticism'] = df_questions[neu].sum(axis=1)
    data_total['Agreeableness'] = df_questions[agr].sum(axis=1)
    data_total['Conscientiousness'] = df_questions[csn].sum(axis=1)
    data_total['Openness'] = df_questions[opn].sum(axis=1)

    st.write(data_total)
    

    # st.write(df_questions[:11].sum(axis=1))

    st.write(predict)


# def create_graph():
#     data_sum = pd.DataFrame()
#     data_sum['Extroversion'] = 


if st.sidebar.button('Analyse'):
    run_algorithm()