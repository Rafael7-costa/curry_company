# importando as bibliotecas
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image
import plotly.express as px

# ---------------------------------------
# 1. FUNÇÕES DE PROCESSAMENTO E LIMPEZA
# ---------------------------------------

def clean_code(df):
    """Limpa o dataframe e converte tipos de dados."""
    df1 = df.copy()
    
    # Remover espaços de colunas string
    for col in df1.select_dtypes(include=['object']).columns:
        df1[col] = df1[col].str.strip()

    # Substituir a string 'NaN' por valor NaN e excluir linhas vazias
    df1.replace('NaN', np.nan, inplace=True)
    df1.dropna(inplace=True)

    # Conversão de tipos
    df1['Delivery_person_Age'] = df1['Delivery_person_Age'].astype(int)
    df1['Delivery_person_Ratings'] = df1['Delivery_person_Ratings'].astype(float)
    df1['Order_Date'] = pd.to_datetime(df1['Order_Date'], format='%d-%m-%Y')
    df1['Vehicle_condition'] = df1['Vehicle_condition'].astype(int)
    
    # Lidar com colunas numéricas
    df1['multiple_deliveries'] = pd.to_numeric(df1['multiple_deliveries'], errors='coerce').fillna(0).astype(int)
    df1['Time_taken(min)'] = df1['Time_taken(min)'].str.replace('(min)', '', regex=False).str.strip().astype(int)

    # Extrair a semana do ano
    df1['Order_Week'] = df1['Order_Date'].dt.isocalendar().week.astype(int)
    
    df1.reset_index(drop=True, inplace=True)
    return df1

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return clean_code(df)

@st.cache_data
def apply_filters(df, date_limit, selected_traffic):
    return df.loc[(df['Order_Date'].dt.date <= date_limit) & 
                  (df['Road_traffic_density'].isin(selected_traffic))].copy()

# ---------------------------------------
# 2. FUNÇÕES PARA MÉTRICAS E TABELAS (OTIMIZAÇÃO)
# ---------------------------------------

def top_delivers(df_filtrado, top_asc):
    """Calcula os entregadores mais rápidos ou mais lentos por cidade."""
    df_avg_time = df_filtrado.groupby(['City', 'Delivery_person_ID'])['Time_taken(min)'].mean().reset_index()
    df_avg_time.rename(columns={'Time_taken(min)': 'Avg_Time_taken(min)'}, inplace=True)
    
    if top_asc:
        # Mais rápidos (menores tempos)
        df_top = df_avg_time.groupby('City').apply(lambda x: x.nsmallest(10, 'Avg_Time_taken(min)')).reset_index(drop=True)
    else:
        # Mais lentos (maiores tempos)
        df_top = df_avg_time.groupby('City').apply(lambda x: x.nlargest(10, 'Avg_Time_taken(min)')).reset_index(drop=True)
    
    return df_top.sort_values(['City', 'Avg_Time_taken(min)'], ascending=[True, top_asc])

def get_ratings_avg_std(df_filtrado, column_name):
    """Calcula média e desvio padrão de avaliações por uma coluna específica."""
    df_ratings = df_filtrado.groupby(column_name)['Delivery_person_Ratings'].agg(['mean', 'std']).reset_index()
    df_ratings.columns = [column_name, 'Avaliação Média', 'Desvio Padrão']
    return df_ratings

# ---------------------------------------
# 3. CONFIGURAÇÃO DO STREAMLIT
# ---------------------------------------

st.set_page_config(layout='wide', page_title="Marketplace - Entregadores", page_icon="🚚")

# Carregamento inicial
df1 = load_data('data_set/train.csv')

# --- SIDEBAR ---
image = Image.open('logo1.png')
st.sidebar.image(image, width=200)

st.sidebar.markdown("# Curry Company")

min_data = df1['Order_Date'].min().date()
max_data = df1['Order_Date'].max().date()

date_slider = st.sidebar.slider('Selecione uma data limite', value=max_data, min_value=min_data, max_value=max_data, format='DD-MM-YYYY')
traffic_selecionado = st.sidebar.multiselect('Condições do trânsito', 
                                             df1['Road_traffic_density'].unique().tolist(),
                                             default=df1['Road_traffic_density'].unique().tolist())

# Aplicação dos filtros no DF de trabalho
df_filtrado = apply_filters(df1, date_slider, traffic_selecionado)

# ---------------------------------------
# 4. LAYOUT DAS ABAS
# ---------------------------------------

st.header('Marketplace - Visão Entregadores')

tab1, tab2, tab3 = st.tabs(['Visão Gerencial', '_', '_'])

with tab1:
    with st.container():
        st.title('Overall Metrics')
        col1, col2, col3, col4 = st.columns(4, gap='large')
        
        # Uso do DF filtrado para métricas dinâmicas
        col1.metric('Maior Idade', df_filtrado['Delivery_person_Age'].max())
        col2.metric('Menor Idade', df_filtrado['Delivery_person_Age'].min())
        col3.metric('Melhor Condição Veicular', df_filtrado['Vehicle_condition'].max())
        col4.metric('Pior Condição Veicular', df_filtrado['Vehicle_condition'].min())

    with st.container():
        st.markdown("""___""")
        st.title('Avaliações')
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Avaliação Média por Entregador')
            media_entregador = df_filtrado[['Delivery_person_ID', 'Delivery_person_Ratings']].groupby('Delivery_person_ID').mean().reset_index()
            st.dataframe(media_entregador)

        with col2:
            st.subheader('Avaliação Média por Trânsito')
            st.dataframe(get_ratings_avg_std(df_filtrado, 'Road_traffic_density'))

            st.subheader('Avaliação Média por Clima')
            st.dataframe(get_ratings_avg_std(df_filtrado, 'Weatherconditions'))

    with st.container():
        st.markdown("""___""")
        st.title('Velocidade de Entrega')
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('TOP 10 Entregadores Mais Rápidos')
            st.dataframe(top_delivers(df_filtrado, top_asc=True))

        with col2:
            st.subheader('TOP 10 Entregadores Mais Lentos')
            st.dataframe(top_delivers(df_filtrado, top_asc=False))























