import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

# pd.options.plotting.backend = "plotly"

from questions_lists import engList

st.write("""
# Five Big Personality Traits

""")

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
data = get_data("data-final.csv")


# @st.cache()
def run_algorithm():
    data_sample = data.sample(n=5000, random_state=1)
    kmeans   = KMeans(n_clusters=5)
    y_kmeans = kmeans.fit(data_sample) # fit_predict

    predict = y_kmeans.labels_
    data_sample['clusters'] = predict

    st.write(data.head(10))

    profile_group = kmeans.predict(user_inputs)[0]
    st.write(f'Meu grupo de personalidade é {profile_group}')

    data_chart = preper_chart_data(data_sample)
    st.write(data_chart.head(10))


def preper_chart_data(data):
    # Selecting columns of each group
    col_list = list(data)
    ext = col_list[:10]
    neu = col_list[10:20]
    agr = col_list[20:30]
    csn = col_list[30:40]
    opn = col_list[40:50]

    data_total = pd.DataFrame()
    data_total['Extroversion'] = data[ext].sum(axis=1)
    data_total['Neuroticism'] = data[neu].sum(axis=1)
    data_total['Agreeableness'] = data[agr].sum(axis=1)
    data_total['Conscientiousness'] = data[csn].sum(axis=1)
    data_total['Openness'] = data[opn].sum(axis=1)
    data_total['clusters'] = data['clusters']
    return data_total


if st.sidebar.button('Analyse'):
    run_algorithm()
    

def analyze_profile(user_inputs):
    profile_group = k_fit.predict(user_inputs)
    st.write('Meu grupo de personalidade é ' + profile_group)


# user_profile_group = analyze_profile(user_inputs)
# st.write(user_profile_group)

st.write(user_inputs)



# st.write(data_chart)

# def load_chart(data_chart):

"""
plt.style.use('bmh')

fig, axs = plt.subplots(figsize=(8,5))

plt.title("")
x = data_chart.columns
y = data_chart.values.T


st.write(x)
st.write(y)


fig = px.bar(data_chart, x=x, y=y)
plt.yticks(fontsize=30)
plt.xticks(fontsize=20, rotation="vertical")
plt.ylim(0, 40)
st.write(fig)

    

data_chart.to_csv("data_chart.csv", index=False)
"""
    # fig, ax = plt.subplots(figsize=(8, 5))
    # st.pyplot(data_total.all())
    
