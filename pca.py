import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import datasets
import datetime

#im = Image.open('C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/Streamlit/Arroz/apps/arroz.png')
im2 = Image.open('r1.jpg')
st.set_page_config(page_title='Cluster-App', layout="wide", page_icon=im2)
st.set_option('deprecation.showPyplotGlobalUse', False)

row1_1, row1_2 = st.columns((2, 3))

with row1_1:
    image = Image.open('r1.jpg')
    st.image(image, use_column_width=True)
    st.markdown('Web App by [Manuel Castiblanco](http://ia.smartecorganic.com.co/index.php/contact/)')
with row1_2:
    st.write("""
    # Reducción Dimensional - PCA App
    Esta App utiliza algoritmos de Machine Learning para hacer hacer reducción dimensional como PCA!
    """)
    with st.expander("Contáctanos 👉"):
        with st.form(key='contact', clear_on_submit=True):
            name = st.text_input('Name')
            mail = st.text_input('Email')
            q = st.text_area("Query")

            submit_button = st.form_submit_button(label='Send')
            if submit_button:
                subject = 'Consulta'
                to = 'macs1251@hotmail.com'
                sender = 'macs1251@hotmail.com'
                smtpserver = smtplib.SMTP("smtp-mail.outlook.com", 587)
                user = 'macs1251@hotmail.com'
                password = '1251macs'
                smtpserver.ehlo()
                smtpserver.starttls()
                smtpserver.ehlo()
                smtpserver.login(user, password)
                header = 'To:' + to + '\n' + 'From: ' + sender + '\n' + 'Subject:' + subject + '\n'
                message = header + '\n' + name + '\n' + mail + '\n' + q
                smtpserver.sendmail(sender, to, message)
                smtpserver.close()

st.header('Aplicación')
st.markdown('____________________________________________________________________')
app_des = st.expander('Descripción App')
with app_des:
    st.write("""Esta aplicación puedes cargar un dataset, en cual puedes validar cual sería la reducción ideal de variables.
    Para que te guies toma un dataset de genética y se hace la reducción con la validación idónea
    Para lo cual debes tene en cuenta:

    - El archivo puede ser cargado en el menú de la parte izquierda
    - EL dataset tiene que estar pre-procesado.
    - Dependiendo del dataset se puede seleccionar el numero de reducción de componentes.
    - El criterio encima de 90% hacia arriba para que la reducción esté bien.
    - Puedes descartar Variables para el análisis.
    - Puedes seleccionar como sería en 3D para tres componentes del dataset a modo de ejemplo, para mejorar el entendimiento.

    """)

st.sidebar.header('Parámetros de Entrada Usario')

# st.sidebar.markdown("""
# [Example CSV input file](penguins_example.csv)
# """)

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Cargue sus parámetros desde un archivo CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv('clean_gen.csv')

st.subheader ('PCA')
st.write('**Tabla de Datos**')
st.dataframe(df.head())
var1=st.multiselect('Variables Desechadas', df.columns, df.columns[0])



with st.expander("Contáctanos👉"):
    st.subheader('Quieres conocer mas de IA, ML o DL 👉[contactanos!!](http://ia.smartecorganic.com.co/index.php/contact/)')
