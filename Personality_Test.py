import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
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
st.markdown("<h1 style='text-align: center'>Teste De Personalidade</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center'>Cinco Grandes Traços De Personalidade</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center'>Abra o menu à esquerda e faça o teste para descobrir a qual grupo de personalidade seu perfil corresponde</h4>", unsafe_allow_html=True)


# @st.cache # load data only at the first time
def get_data(filename):
    data = pd.read_csv(filename)
    return data


# st.sidebar.header('Forneça as respostas')



def user_input_features(qlist):
    questionsDict = {}
    for q in qlist:
        questionsDict[q] = st.sidebar.slider(q, 1, 5, 3)
    questionsDF = pd.DataFrame(questionsDict, index=[0])
    return questionsDF


with st.sidebar:
    form = st.form(key='slider_form') #creating the form
    submit_button = form.form_submit_button(label='Analisar Perfil')
    with form: #creating the form
        user_inputs =  user_input_features(port_list)


# Carrega modelo treinado
model = load("model.joblib")

# data = get_data("data-final.csv")

data_chart_groups = get_data("data_chart_groups.csv")
data_chart_groups.columns = ["clusters","Extroversão","Neuroticismo","Amabilidade","Consciência","Abertura"]



# Gráfico de linhas com plotly
fig = go.Figure()
for i in range(5):
    fig.add_trace(go.Scatter(x=data_chart_groups.columns[1:], y=list(data_chart_groups.iloc[i, 1:]), mode='lines', name=f'Grupo {i+1}'))
fig.update_layout(
    title="",
    font=dict(size=15),
    autosize=False,
    height=260,
    #width=int(screen_width)*.36,
    xaxis_title="", 
    margin=dict(l=0, r=0, t=30, b=20)
)
fig.update_yaxes(
    tickvals=[2, 2.5, 3, 3.5, 4],
    range=[2.3, 3.8]
)
st.plotly_chart(fig,use_container_width=True)
# st.write(fig)


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
    st.markdown(f"<h2 style='text-align: center'>{group_ident}</h2>", unsafe_allow_html=True)


    # Gráfico de área do perfil corespondente
    fig = px.area(x=list(data_chart_groups.columns[1:]), y=list(data_chart_groups.iloc[profile_group][1:]))
    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=20),
        font=dict(size=15),
        xaxis_title="", yaxis_title="",
        autosize=False,
        height=300,
        #width=int(screen_width)*.36,
    )
    fig.update_yaxes(
        tickvals=[2, 2.5, 3, 3.5, 4],
        range=[2.3, 3.8]
    )
    # fig.suptitle(f'Profile Group: {profile_group}')
    # st.write(fig)
    
    st.plotly_chart(fig,use_container_width=True)
    
    st.sidebar.text("Análise concluída!")


# st.form_submit_button returns True upon form submit
if submit_button:
    run_algorithm()




# components.html("""
#     <p style="text-align: justify; font-family:sans-serif; font-size: 1rem; font-weight: 400; color: rgb(38, 39, 48); line-height: 1.6">
obs = ("""Todos esses traços de personalidade estão presentes em cada pessoa. 
    O gráfico de linhas acima mostra cinco grupos de pessoas, de acordo 
    com cada nível dos traços de personalidade. Responda às perguntas na barra lateral 
    para ver a qual grupo corresponde ao seu perfil.""")
st.write(f"<p style='text-align: justify'>{obs}</p>", unsafe_allow_html=True)
    # </p>""", height=100)
st.markdown("""---""")




# Definindo colunas do layout
col1, col2 = st.beta_columns([1,3])

col1.image("img/extroversion.png")
with col2:
    extroversion = ("""
        (Extroversão) - Pessoas com alto índice de extroversão são extrovertidas e tendem a ganhar 
        energia em situações sociais. Estar perto de outras pessoas os ajuda a se sentirem 
        energizados e animados. Pessoas com baixa extroversão (ou introvertidas) tendem a 
        ser mais reservadas e têm menos energia para gastar em ambientes sociais. Os eventos 
        sociais podem parecer exaustivos e os introvertidos costumam exigir um período de 
        solidão e sossego para 'recarregar as energias'.
        """)
    col2.write(f"<p style='text-align: justify'>{extroversion}</p>", unsafe_allow_html=True)
    
st.markdown("""---""")
col1, col2 = st.beta_columns([1,3])


col1.image("img/neuroticism.png")
with col2:
    neuroticism = ("""(Neuroticismo) - Neuroticismo é um traço caracterizado por tristeza, 
        mau humor e instabilidade emocional. Indivíduos com alto teor dessa característica 
        tendem a sofrer oscilações de humor, ansiedade, irritabilidade e tristeza. Aqueles 
        com baixo nível dessa característica tendem a ser mais estáveis ​​e emocionalmente 
        resilientes.
        """)
    col2.write(f"<p style='text-align: justify'>{neuroticism}</p>", unsafe_allow_html=True)

st.markdown("""---""")
col1, col2 = st.beta_columns([1,3])


col1.image("img/agreeableness.png")
with col2:
    agreeableness = ("""
        (Amabilidade) - Essa dimensão da personalidade inclui atributos como confiança, 
        altruísmo, gentileza, afeto e outros comportamentos pró-sociais. Pessoas com alto 
        nível de gentileza tendem a ser mais cooperativas, enquanto aqueles com baixo nível 
        desse traço tendem a ser mais competitivos e às vezes até manipuladores.
        """)
    col2.write(f"<p style='text-align: justify'>{agreeableness}</p>", unsafe_allow_html=True)

st.markdown("""---""")
col1, col2 = st.beta_columns([1,3])

col1.image("img/conscientiousness.png")
with col2:
    conscientiousness = ("""
        (Consciência) - As características padrão dessa dimensão incluem altos níveis de 
        consideração, bom controle de impulsos e comportamentos direcionados a objetivos. 
        Pessoas altamente conscienciosas tendem a ser organizadas e atentas aos detalhes. 
        Eles planejam com antecedência, pensam sobre como seu comportamento afeta os outros e 
        estão atentos aos prazos.
        """)
    col2.write(f"<p style='text-align: justify'>{conscientiousness}</p>", unsafe_allow_html=True)

st.markdown("""---""")
col1, col2 = st.beta_columns([1,3])

col1.image("img/openness.png")
with col2:
    openness = (f"""
        (Abertura) - Esse traço apresenta características como imaginação e percepção.
        Pessoas com alto valor nesse traço também tendem a ter uma ampla gama de interesses. 
        Eles são curiosos sobre o mundo e outras pessoas e estão ansiosos para aprender coisas 
        novas e desfrutar de novas experiências. Pessoas com alto nível desse atributo tendem a 
        ser mais aventureiras e criativas. Pessoas com baixo nível dessa característica costumam 
        ser muito mais tradicionais e podem ter dificuldades com o pensamento abstrato.
        """)
    col2.write(f"<p style='text-align: justify'>{openness}</p>", unsafe_allow_html=True)

st.write("fonte: https://www.verywellmind.com/")
