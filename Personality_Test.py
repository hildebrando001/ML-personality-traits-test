import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


from questions_lists import engList

st.write("""
# Five Big Personality Traits

""")

@st.cache # load data only at the first time
def get_data(filename):
    data = pd.read_csv(filename)
    return data


st.sidebar.header('Answer the questions')


def user_input_features(qlist):
    questionsDict = {}
    for q in qlist:
        questionsDict[q] = st.sidebar.slider(q, 1, 5, 3)
    questionsDF = pd.DataFrame(questionsDict, index=[0])
    return questionsDF

df_questions = user_input_features(engList)
data = get_data("data-final.csv")


@st.cache()
def run_algorithm():
    kmeans = KMeans(n_clusters=5)
    k_fit = kmeans.fit(data)
    predict = k_fit.predict(df_questions)


if st.sidebar.button('Analyse'):
    run_algorithm()


def preper_chart_data(df_questions):
    # Selecting columns of each group
    col_list = list(df_questions)
    ext = col_list[:10]
    neu = col_list[10:20]
    agr = col_list[20:30]
    csn = col_list[30:40]
    opn = col_list[40:50]

    data_total = pd.DataFrame()
    data_total['Extroversion'] = df_questions[ext].sum(axis=1)/5
    data_total['Neuroticism'] = df_questions[neu].sum(axis=1)/5
    data_total['Agreeableness'] = df_questions[agr].sum(axis=1)/5
    data_total['Conscientiousness'] = df_questions[csn].sum(axis=1)/5
    data_total['Openness'] = df_questions[opn].sum(axis=1)/5
    return data_total


data_total = preper_chart_data(df_questions)

def load_chart(data):
    plt.style.use('bmh')
    fig, axs = plt.subplots(figsize=(5,5))
    plt.title("")
    plt.barh(data.columns, data.iloc[:, 0].sort_values(ascending=False))
    plt.yticks(fontsize=16)
    plt.xticks(fontsize=0)
    # plt.xlim(0, data.iloc[:, 0].max() + 1)
    st.write(fig)
    
load_chart(data_total)

# st.write(data_total)

    # data_total.to_csv("data_total.csv", index=False)

    # fig, ax = plt.subplots(figsize=(8, 5))
    # st.pyplot(data_total.all())
    
