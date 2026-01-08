# importando as bibliotecas
from haversine import haversine
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from PIL import Image
import folium 
from streamlit_folium import folium_static 

# ---------------------------------------
# Funções de Processamento
# ---------------------------------------

def clean_code(df1):
    """Limpa o dataframe e converte tipos de dados."""
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
    
    # Lidar com valores não numéricos e NaN na coluna 'multiple_deliveries'
    df1['multiple_deliveries'] = pd.to_numeric(df1['multiple_deliveries'], errors='coerce').fillna(0).astype(int)
    
    # Remover "(min)" e espaços da coluna 'Time_taken(min)'
    df1['Time_taken(min)'] = df1['Time_taken(min)'].str.replace('(min)', '', regex=False).str.strip().astype(int)
    
    # Extrair a semana do ano
    df1['Order_Week'] = df1['Order_Date'].dt.isocalendar().week.astype(int)
    
    df1.reset_index(drop=True, inplace=True)
    return df1

@st.cache_data
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    return clean_code(df)

@st.cache_data
def apply_filters(df, date_limit, selected_traffic):
    df_filtrado = df.loc[
        (df['Order_Date'].dt.date <= date_limit) & 
        (df['Road_traffic_density'].isin(selected_traffic))
    ].copy() 
    return df_filtrado

# ---------------------------------------
# Funções de Gráficos (Otimização)
# ---------------------------------------

def order_metric_chart(df_filtrado):
    daily_orders = df_filtrado.groupby('Order_Date').size().reset_index(name='Total_Pedidos')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Order_Date', y='Total_Pedidos', data=daily_orders, color='#87CEEB', ax=ax) 
    ax.set_title('Quantidade de Pedidos por Dia')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def traffic_order_share(df_filtrado):
    traffic_dist = df_filtrado['Road_traffic_density'].value_counts().reset_index()
    traffic_dist.columns = ['Tipo de Tráfego', 'Total_Pedidos']
    fig = px.pie(traffic_dist, values='Total_Pedidos', names='Tipo de Tráfego', hole=0.3,
                 color_discrete_sequence=px.colors.sequential.RdBu)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def city_traffic_bubble(df_filtrado):
    city_traffic = df_filtrado.groupby(['City', 'Road_traffic_density']).size().reset_index(name='Total_Pedidos')
    fig = px.scatter(city_traffic, x='City', y='Road_traffic_density', size='Total_Pedidos', 
                     color='City', title='Volume por Cidade e Tráfego', template='plotly_white')
    return fig

def weekly_order_line(df_filtrado):
    weekly_orders = df_filtrado.groupby('Order_Week').size().reset_index(name='Total_Pedidos')
    fig = px.line(weekly_orders, x='Order_Week', y='Total_Pedidos', markers=True, 
                  color_discrete_sequence=['#87CEEB'], template='plotly_white')
    return fig

def avg_orders_per_delivery_person(df_filtrado):
    delivery_weekly = df_filtrado.groupby(['Delivery_person_ID', 'Order_Week']).size().reset_index(name='Total')
    avg_weekly = delivery_weekly.groupby('Order_Week')['Total'].mean().reset_index()
    fig = px.line(avg_weekly, x='Order_Week', y='Total', markers=True, 
                  color_discrete_sequence=['#2E8B57'], template='plotly_white')
    return fig

def country_map(df_filtrado):
    data_plot = df_filtrado.loc[:, ['City', 'Road_traffic_density', 'Delivery_location_latitude', 'Delivery_location_longitude']].dropna()
    data_plot = data_plot.head(50) # Limitando para performance no Folium
    
    map_ = folium.Map(location=[data_plot['Delivery_location_latitude'].mean(), 
                                data_plot['Delivery_location_longitude'].mean()], zoom_start=11)
    
    for _, row in data_plot.iterrows():
        folium.Marker([row['Delivery_location_latitude'], row['Delivery_location_longitude']], 
                      popup=f"{row['City']} - {row['Road_traffic_density']}").add_to(map_)
    return map_

# ---------------------------------------
# Início da Estrutura Lógica do App
# ---------------------------------------

st.set_page_config(layout='wide', page_title="Marketplace - Visão Cliente", page_icon="🛒")

# Carregar dados
try:
    df_raw = load_and_clean_data('data_set/train.csv')
except Exception as e:
    st.error(f"Erro ao carregar arquivo: {e}")
    st.stop()

# --- SIDEBAR ---
image = Image.open('logo1.png')
st.sidebar.image(image, width=200)

st.sidebar.markdown("# Curry Company")

min_date = df_raw['Order_Date'].min().date()
max_date = df_raw['Order_Date'].max().date()

date_slider = st.sidebar.slider('Selecione uma data limite', value=max_date, min_value=min_date, max_value=max_date, format='DD-MM-YYYY')

traffic_options = st.sidebar.multiselect('Condições do trânsito', 
                                         df_raw['Road_traffic_density'].unique().tolist(),
                                         default=df_raw['Road_traffic_density'].unique().tolist())

# Aplicação dos Filtros
df_filtrado = apply_filters(df_raw, date_slider, traffic_options)

# --- LAYOUT PRINCIPAL ---
st.header('Marketplace - Visão Cliente')

if df_filtrado.empty:
    st.warning("Ajuste os filtros, não há dados para essa seleção.")
    st.stop()

tab1, tab2, tab3 = st.tabs(['Visão Gerencial', 'Visão Tática', 'Visão Geográfica'])

with tab1:
    with st.container():
        st.markdown ("### Quantidade de Pedidos por Dia")
        st.pyplot(order_metric_chart(df_filtrado))
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Distribuição por Tráfego")
        st.plotly_chart(traffic_order_share(df_filtrado), use_container_width=True)
    with col2:
        st.markdown("### Volume por Cidade e Tráfego")
        st.plotly_chart(city_traffic_bubble(df_filtrado), use_container_width=True)

with tab2:
    st.markdown("### Pedidos por Semana")
    st.plotly_chart(weekly_order_line(df_filtrado), use_container_width=True)
    
    st.markdown("### Média de Pedidos por Entregador/Semana")
    st.plotly_chart(avg_orders_per_delivery_person(df_filtrado), use_container_width=True)

with tab3:
    st.markdown("### Visualização Geográfica")
    mapa = country_map(df_filtrado)
    folium_static(mapa)