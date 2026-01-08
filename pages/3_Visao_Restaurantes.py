# importando as bibliotecas
from haversine import haversine, Unit
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# ------------------------------------
# 1. FUNÇÕES DE PROCESSAMENTO
# ------------------------------------

def clean_code(df):
    df1 = df.copy()
    for col in df1.select_dtypes(include=['object']).columns:
        df1[col] = df1[col].str.strip()
    df1.replace('NaN', np.nan, inplace=True)
    df1.dropna(inplace=True)
    
    df1['Delivery_person_Age'] = df1['Delivery_person_Age'].astype(int)
    df1['Delivery_person_Ratings'] = df1['Delivery_person_Ratings'].astype(float)
    df1['Order_Date'] = pd.to_datetime(df1['Order_Date'], format='%d-%m-%Y')
    df1['multiple_deliveries'] = pd.to_numeric(df1['multiple_deliveries'], errors='coerce').fillna(0).astype(int)
    df1['Time_taken(min)'] = df1['Time_taken(min)'].str.replace('(min)', '', regex=False).str.strip().astype(int)
    df1['Order_Week'] = df1['Order_Date'].dt.isocalendar().week.astype(int)
    
    # Cálculo de distância médio global inicial
    df1['distance_km'] = df1.apply(lambda x: haversine(
        (x['Restaurant_latitude'], x['Restaurant_longitude']), 
        (x['Delivery_location_latitude'], x['Delivery_location_longitude']), unit=Unit.KILOMETERS), axis=1)
    
    return df1

@st.cache_data
def load_data(file_path):
    return clean_code(pd.read_csv(file_path))

# ------------------------------------
# 2. FUNÇÕES DE GRÁFICOS E MÉTRICAS
# ------------------------------------

def festival_metrics(df_filtrado, festival_status, metric_type):
    """Retorna média ou desvio padrão baseado no status do festival."""
    df_aux = df_filtrado.groupby('Festival')['Time_taken(min)'].agg(['mean', 'std']).reset_index()
    df_aux = df_aux.loc[df_aux['Festival'] == festival_status, metric_type]
    return f"{df_aux.iloc[0]:.2f}" if not df_aux.empty else "0.00"

def distance_pie_chart(df_filtrado):
    """Gráfico de pizza da distância média por tipo de cidade."""
    df_aux = df_filtrado.groupby('City')['distance_km'].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(df_aux['distance_km'], labels=df_aux['City'], autopct='%1.1f%%', startangle=90, shadow=True)
    ax.set_title('Distância Média por Cidade')
    return fig

def city_time_bar_chart(df_filtrado):
    """Gráfico de barras de tempo médio com desvio padrão."""
    df_aux = df_filtrado.groupby('City')['Time_taken(min)'].agg(['mean', 'std']).reset_index()
    df_aux.columns = ['City', 'Avg_Time', 'Std_Time']
    fig = px.bar(df_aux, x='City', y='Avg_Time', error_y='Std_Time', color='City',
                 title='Tempo Médio de Entrega por Cidade', labels={'Avg_Time': 'Tempo (min)'})
    return fig

def sunburst_delivery_chart(df_filtrado):
    """Gráfico Sunburst para Cidade e Tráfego."""
    df_aux = df_filtrado.groupby(['City', 'Road_traffic_density'])['Time_taken(min)'].agg(['mean', 'std']).reset_index()
    df_aux.columns = ['City', 'Traffic', 'Avg_Time', 'Std_Time']
    fig = px.sunburst(df_aux, path=['City', 'Traffic'], values='Avg_Time', color='Std_Time',
                      color_continuous_scale='Reds', title='Hierarquia: Cidade > Tráfego')
    return fig

# ------------------------------------
# 3. CONFIGURAÇÃO E FILTROS
# ------------------------------------

st.set_page_config(layout='wide', page_title="Marketplace - Restaurantes",  page_icon="🍔")
df1 = load_data('data_set/train.csv')

# Sidebar
image = Image.open('logo1.png')
st.sidebar.image(image, width=200)

st.sidebar.markdown("# Curry Company")
date_slider = st.sidebar.slider('Selecione uma data limite', 
                                value=df1['Order_Date'].max().date(),
                                min_value=df1['Order_Date'].min().date(),
                                max_value=df1['Order_Date'].max().date(), format='DD-MM-YYYY')
traffic_options = st.sidebar.multiselect('Trânsito', df1['Road_traffic_density'].unique().tolist(), 
                                         default=df1['Road_traffic_density'].unique().tolist())

df_filtrado = df1.loc[(df1['Order_Date'].dt.date <= date_slider) & (df1['Road_traffic_density'].isin(traffic_options))].copy()

# ------------------------------------
# 4. LAYOUT
# ------------------------------------

st.header('Marketplace - Visão Restaurantes')

tab1, tab2, tab3 = st.tabs(['Visão Gerencial', '_', '_'])

with tab1:
    with st.container():
        st.subheader('Overall Metrics')
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric('Entregadores Únicos', df_filtrado['Delivery_person_ID'].nunique())
        c2.metric('Distância Média', f"{df_filtrado['distance_km'].mean():.2f} km")
        c3.metric('T. Médio (Festival)', festival_metrics(df_filtrado, 'Yes', 'mean'))
        c4.metric('DP Médio (Festival)', festival_metrics(df_filtrado, 'Yes', 'std'))
        c5.metric('T. Médio (Sem Fest)', festival_metrics(df_filtrado, 'No', 'mean'))
        c6.metric('DP Médio (Sem Fest)', festival_metrics(df_filtrado, 'No', 'std'))

    st.markdown("---")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(distance_pie_chart(df_filtrado))
        with col2:
            st.subheader('Tempo e DP por Cidade e Pedido')
            df_aux = df_filtrado.groupby(['City', 'Type_of_order'])['Time_taken(min)'].agg(['mean', 'std']).reset_index()
            st.dataframe(df_aux)

    st.markdown("---")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(city_time_bar_chart(df_filtrado), use_container_width=True)
        with col2:
            st.plotly_chart(sunburst_delivery_chart(df_filtrado), use_container_width=True)












