import streamlit as st 
from PIL import Image

st.set_page_config(
    page_title="Home" 
)

#image_path = 'logo1.png'
image = Image.open('logo1.png')
st.sidebar.image(image, width=200)

st.sidebar.markdown("# Curry Company") 
st.sidebar.markdown('## Fastest Delivery in Town')
st.sidebar.markdown("""---""")

st.write("# Curry Company Growth Dashboard")

st.markdown(
    """
    Growth Dashboard foi construído para acompanhar métricas de crescimento dos Entregadores e Restaurantes.
    ### Como utilizar esse Growth Dashboard?
    - **Visão Empresa:**
        - Visão Gerencial: Métricas gerais de comportamento.
        - Visão Tática: Indicadores semanais de crescimento.
        - Visão Geográfica: Insights de geolocalização.
    - **Visão Entregador:**
        - Acompanhamento dos indicadores semanais de crescimento.
    - **Visão Restaurante:**
        - Indicadores semanais de crescimento dos restaurantes.
    ### Ask for Help
    - Time de dados
""")