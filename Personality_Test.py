import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit.components.v1 as components

# Salva modelo treinado em arquivo no disco
from joblib import dump, load

# 
from questions_lists import engList

# st.write("""
# # Five Big Personality Traits
# """)
st.markdown("<h1 style='text-align: center'>Five Big Personality Traits</h1>", unsafe_allow_html=True)

# @st.cache # load data only at the first time
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


user_inputs = user_input_features(engList)

# Carrega modelo treinado
model = load("model.joblib")

# data = get_data("data-final.csv")

data_chart_groups = get_data("data_chart_groups.csv")

# Gráfico de linhas com seaborn
sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(figsize=(12, 3))
for i in range(5):
    ax = sns.lineplot(data=data_chart_groups, x=data_chart_groups.columns[1:], y=data_chart_groups.iloc[i, 1:])
plt.ylabel("")
plt.xticks(rotation=0, size=(14))
# plt.ylim(2, 4)
plt.legend(['Group 1', 'Group 2', 'Group 3', 'Group 4', 'Group 5'])
st.write(fig)


st.write("""
    All these personality traits are present in each people. The line chart above shows five groups of people, according to the personalty trait level. Answer the questions to see in which group you are.
""")
st.markdown("""---""")


# @st.cache()
def run_algorithm():
    # Analisa predição em toda base de dados
    # data_sample = data.sample(n=5000, random_state=1) # volume de dados reduzidos para teste
    # kmeans   = KMeans(n_clusters=5, random_state=0, max_iter=300).fit(data)
    # y_kmeans = kmeans.fit(data) já rodou na linha acima

    # Utilizando o joblib para gerar arquivo com modelo treinado
    # dump(kmeans, 'model.joblib')

    # Adiciona os respectivos grupos a cada linha do dataset
    # predict = kmeans.labels_
    # data['clusters'] = predict

    # Compara os dados fornecidos pelo usuário aos perfis esbalecidos anteriormente
    profile_group = model.predict(user_inputs)[0] # kmeans
    
    group_ident = f'Personality corresponding to Group {profile_group + 1}'
    st.markdown(f"<h1 style='text-align: center'>{group_ident}</h1>", unsafe_allow_html=True)


    # Gera o gráfico do perfil correspondente
    fig = px.bar(data_chart_groups, x=data_chart_groups.columns[1:], y=list(data_chart_groups.iloc[profile_group][1:]))
    # fig.suptitle(f'Profile Group: {profile_group}')
    st.write(fig)


if st.sidebar.button('Analyse'):
    run_algorithm()



# Definindo colunas do layout
col1, col2 = st.beta_columns([1,3])

col1.image("img/extroversion.png")
col2.write("""
    (Extroversion) - People who are high in extraversion are outgoing and tend to gain energy in social situations. Being around other people helps them feel energized and excited.
    People who are low in extraversion (or introverted) tend to be more reserved and have less energy to expend in social settings. Social events can feel draining and introverts often require a period of solitude and quiet in order to 'recharge.'
""")


st.markdown("""---""")
col1, col2 = st.beta_columns([1,3])


col1.image("img/neuroticism.png")
col2.write("""
    (Neuroticism) - Neuroticism is a trait characterized by sadness, moodiness, and emotional instability.
    Individuals who are high in this trait tend to experience mood swings, anxiety, irritability, and sadness. Those low in this trait tend to be more stable and emotionally resilient.
""")

st.markdown("""---""")
col1, col2 = st.beta_columns([1,3])

col1.image("img/agreeableness.png")
col2.write("""
    (Agreeableness) - This personality dimension includes attributes such as trust, altruism, kindness, affection, and other prosocial behaviors.﻿ People who are high in agreeableness tend to be more cooperative while those low in this trait tend to be more competitive and sometimes even manipulative.
""")

st.markdown("""---""")
col1, col2 = st.beta_columns([1,3])

col1.image("img/conscientiousness.png")
col2.write("""
    (Conscientiousness) - Standard features of this dimension include high levels of thoughtfulness, good impulse control, and goal-directed behaviors.﻿ Highly conscientious people tend to be organized and mindful of details. They plan ahead, think about how their behavior affects others, and are mindful of deadlines.
""")

st.markdown("""---""")
col1, col2 = st.beta_columns([1,3])

col1.image("img/openness.png")
col2.write("""
    This trait features characteristics such as imagination and insight. People who are high in this trait also tend to have a broad range of interests. They are curious about the world and other people and eager to learn new things and enjoy new experiences.
    People who are high in this trait tend to be more adventurous and creative. People low in this trait are often much more traditional and may struggle with abstract thinking.
""")




