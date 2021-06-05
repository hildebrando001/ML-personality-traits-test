import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit.components.v1 as components
import plotly.graph_objects as go

# Salva modelo treinado em arquivo no disco
from joblib import dump, load

# 
from questions_lists import eng_list, port_list

# st.write("""
# # Five Big Personality Traits
# """)
st.markdown("<h1 style='text-align: center'>Cinco Grandes Traços De Personalidade</h1>", unsafe_allow_html=True)

# @st.cache # load data only at the first time
def get_data(filename):
    data = pd.read_csv(filename)
    return data


st.sidebar.header('Forneça as respostas')
st.sidebar.text('(Deslise o ponto para responder)')

def user_input_features(qlist):
    questionsDict = {}
    for q in qlist:
        questionsDict[q] = st.sidebar.slider(q, 1, 5, 3)
    questionsDF = pd.DataFrame(questionsDict, index=[0])
    return questionsDF


user_inputs = user_input_features(port_list)

# Carrega modelo treinado
model = load("model.joblib")

# data = get_data("data-final.csv")

data_chart_groups = get_data("data_chart_groups.csv")
data_chart_groups.columns = ["clusters","Extroversão","Neuroticismo","Amabilidade","Consciência","Abertura"]

# Gráfico de linhas com seaborn
# sns.set_theme(style="darkgrid")
# fig, ax = plt.subplots(figsize=(12, 3))
# for i in range(5):
#     ax = sns.lineplot(data=data_chart_groups, x=data_chart_groups.columns[1:], y=data_chart_groups.iloc[i, 1:])
# plt.ylabel("")
# plt.xticks(rotation=0, size=(14))
# # plt.ylim(2, 4)
# plt.legend(['Grupo 1', 'Grupo 2', 'Grupo 3', 'Grupo 4', 'Grupo 5'])
# st.write(fig)

# Gráfico de linhas com plotly
fig = go.Figure()
for i in range(5):
    fig.add_trace(go.Scatter(x=data_chart_groups.columns[1:], y=list(data_chart_groups.iloc[i, 1:]), mode='lines', name=f'Grupo {i+1}'))
fig.update_layout(
        font=dict(
            size=15
        )
    )
st.write(fig)


st.write("""
    Todos esses traços de personalidade estão presentes em cada pessoa. 
    O gráfico de linhas acima mostra cinco grupos de pessoas, de acordo 
    com seu nível de traço de personalidade. Responda às perguntas na barra lateral 
    para ver a qual grupo corresponde ao seu perfil.
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
    
    group_ident = f'Seu perfil corresponde ao Grupo {profile_group + 1}'
    st.markdown(f"<h1 style='text-align: center'>{group_ident}</h1>", unsafe_allow_html=True)


    # Gera o gráfico do perfil correspondente
    fig = px.bar(data_chart_groups, x=data_chart_groups.columns[1:], y=list(data_chart_groups.iloc[profile_group][1:]))
    fig.update_layout(
        font=dict(
            size=15
        )
    )
   

    # fig.suptitle(f'Profile Group: {profile_group}')
    st.write(fig)




if st.sidebar.button('Analisar Perfil'):
    run_algorithm()



# Definindo colunas do layout
col1, col2 = st.beta_columns([1,3])

col1.image("img/extroversion.png")
col2.write("""
    (Extroversão) - Pessoas com alto índice de extroversão são extrovertidas e tendem a ganhar energia em situações sociais. Estar perto de outras pessoas os ajuda a se sentirem energizados e animados.
Pessoas com baixa extroversão (ou introvertidas) tendem a ser mais reservadas e têm menos energia para gastar em ambientes sociais. Os eventos sociais podem parecer exaustivos e os introvertidos costumam exigir um período de solidão e sossego para "recarregar as energias".
""")


st.markdown("""---""")
col1, col2 = st.beta_columns([1,3])


col1.image("img/neuroticism.png")
col2.write("""
    (Neuroticismo) - Neuroticismo é um traço caracterizado por tristeza, mau humor e instabilidade emocional.
Indivíduos com alto teor dessa característica tendem a sofrer oscilações de humor, ansiedade, irritabilidade e tristeza. Aqueles com baixo nível dessa característica tendem a ser mais estáveis ​​e emocionalmente resilientes.
""")

st.markdown("""---""")
col1, col2 = st.beta_columns([1,3])

col1.image("img/agreeableness.png")
col2.write("""
    (Amabilidade) - Essa dimensão da personalidade inclui atributos como confiança, altruísmo, gentileza, afeto e outros comportamentos pró-sociais. Pessoas com alto nível de gentileza tendem a ser mais cooperativas, enquanto aqueles com baixo nível desse traço tendem a ser mais competitivos e às vezes até manipuladores.
""")

st.markdown("""---""")
col1, col2 = st.beta_columns([1,3])

col1.image("img/conscientiousness.png")
col2.write("""
    (Consciência) - As características padrão dessa dimensão incluem altos níveis de consideração, bom controle de impulsos e comportamentos direcionados a objetivos. Pessoas altamente conscienciosas tendem a ser organizadas e atentas aos detalhes. Eles planejam com antecedência, pensam sobre como seu comportamento afeta os outros e estão atentos aos prazos.
""")

st.markdown("""---""")
col1, col2 = st.beta_columns([1,3])

col1.image("img/openness.png")
col2.write("""
    (Abertura) - Esse traço apresenta características como imaginação e percepção.
Pessoas com alto valor nesse traço também tendem a ter uma ampla gama de interesses. Eles são curiosos sobre o mundo e outras pessoas e estão ansiosos para aprender coisas novas e desfrutar de novas experiências.
Pessoas com alto nível desse atributo tendem a ser mais aventureiras e criativas. Pessoas com baixo nível dessa característica costumam ser muito mais tradicionais e podem ter dificuldades com o pensamento abstrato.
""")

st.write("fonte: https://www.verywellmind.com/")
