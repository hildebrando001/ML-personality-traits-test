# Como o csv é muito grande e devido a dificuldade em enviar-lo para o github
# eu resolvi utilizar o joblib para gerar um modelo já treinado.
# Isto fará, também, com que o app em streamlit rode mais rápido. Não tendo que
# retreinar o modelo a cada execução.

import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans

# Salva modelo treinado em arquivo no disco
from joblib import dump # , load


st.write("""
# Five Big Personality Traits
## *Creates persistent file using joblib
## *Creates data_chart.csv

""")

# @st.cache # load data only at the first time
def get_data(filename):
    data = pd.read_csv(filename)
    return data


data = get_data("data-final.csv")


# @st.cache()
def run_algorithm():
    # Analisa predição em toda base de dados
    # data_sample = data.sample(n=5000, random_state=1) # volume de dados reduzidos para teste
    kmeans   = KMeans(n_clusters=5, random_state=0, max_iter=300).fit(data)
    # y_kmeans = kmeans.fit(data) já rodou na linha acima

    # Utilizando o joblib para gerar arquivo com modelo treinado
    dump(kmeans, 'model.joblib')


    # Adiciona os respectivos grupos a cada linha do dataset "clusters"
    predict = kmeans.labels_
    data['clusters'] = predict


    # Chama a função para gerar os dados dos gráficos
    preper_chart_data(data)


def preper_chart_data(data):
    # Selecting columns of each group
    col_list = list(data)
    ext = col_list[:10]
    neu = col_list[10:20]
    agr = col_list[20:30]
    csn = col_list[30:40]
    opn = col_list[40:50]

    data_chart = pd.DataFrame()
    data_chart['Extroversion'] = data[ext].sum(axis=1)/10
    data_chart['Neuroticism'] = data[neu].sum(axis=1)/10
    data_chart['Agreeableness'] = data[agr].sum(axis=1)/10
    data_chart['Conscientiousness'] = data[csn].sum(axis=1)/10
    data_chart['Openness'] = data[opn].sum(axis=1)/10
    data_chart['clusters'] = data['clusters']

    data_chart.to_csv("data_chart.csv", index=False)

    st.write(""" # Done""")


if st.button('Create files'):
    run_algorithm()
