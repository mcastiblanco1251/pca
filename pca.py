import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import datasets
import datetime

#im = Image.open('C:/Users/Mcastiblanco/Documents/AGPC/DataScience2020/Streamlit/Arroz/apps/arroz.png')
im2 = Image.open('r1.jpg')
st.set_page_config(page_title='Cluster-App', layout="wide", page_icon=im2)
#st.set_option('deprecation.showPyplotGlobalUse', False)

row1_1, row1_2 = st.columns((2, 3))

with row1_1:
    image = Image.open('r1.jpg')
    st.image(image, use_container_width=True)
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
    - Asegurarse que el archivo tenga la variable a predecir y que esté en la última columna (ver dataset de ejemplo de la app)
    - Dependiendo del dataset se puede seleccionar el numero de reducción de componentes.
    - La app calcula que el 95% de la variación esta dada por el n de componentes.
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
    st.write('A la espera de que se cargue el archivo CSV. Actualmente usando parámetros de entrada de ejemplo (que se muestran a continuación).')
    st.write(df[:1])

st.subheader ('PCA')
st.write('**Tabla de Datos Incial**')
st.dataframe(df.head())
var1=st.multiselect('Variables a Desechar si Aplica', df.columns, df.columns[0])
df.drop(var1,axis=1, inplace=True)
st.write('**Tabla de Datos Procesada**')
st.dataframe(df.head())
st.write(f'**Numero de Variables Iniciales Totales: {df.shape[1]-1}**')

for i in range (df.shape[1]):
    X_std = StandardScaler().fit_transform(df.drop(df.iloc[:,-1:],axis=1))
    sklearn_pca = sklearnPCA(n_components=i)
    Y_sklearn = sklearn_pca.fit_transform(X_std)
    cum_sum = sklearn_pca.explained_variance_ratio_.cumsum()
    v=sklearn_pca.explained_variance_ratio_[:i].sum()
    cum_sum = cum_sum*100
    if v>0.95:
        break
st.subheader('Resultados')

row1_3, row1_4 = st.columns((3, 2))
with row1_3:
    st.write(f'**Varianza alcanzada: {round(v*100,2)}%**')
    st.write(f'**Número de variables ideal dados por el algoritmo: {i}**')

with row1_4:
    fig, ax = plt.subplots(figsize=(8,6))
    plt.bar(range(i), cum_sum, label='Suma acumulada de la varianza', color = 'b',alpha=0.5)
    plt.title(f'Alerededor {round(v*100,2)}% de la varianza es explicada por las primeras {i} variables')
    plt.show()
    st.pyplot(fig)

st.write('**Nuevo Dataset para Usar**')
df2=pd.DataFrame(Y_sklearn)
df2['cat'] =  df['cat'].reset_index().cat
st.dataframe(df2.head())
csv=df2.to_csv().encode('utf-8')

#csv
st.download_button(
 label="Descargar Datos como CSV",
 data=csv,
 file_name='PCA.csv',
)
st.subheader('Si quieres ver como trabaja el PCA de 3D a 2D sigue..')
if st.checkbox('Complementar Visual de Reducción PCA 3D a 2D',value=False):

    row1_5, row1_6=st.columns((2,2))
    with row1_5:
        sklearn_pca = sklearnPCA(n_components=3)
        X_reduced  = sklearn_pca.fit_transform(X_std)
        Y=df2.iloc[:,-1:]
        from mpl_toolkits.mplot3d import Axes3D
        plt.clf()
        fig = plt.figure(1, figsize=(10,6 ))
        ax = Axes3D(fig, elev=-150, azim=110,)
        ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=Y,cmap=plt.cm.Paired,linewidths=10)
        ax.set_title("Primeros 3 direcciones de PCA")
        ax.set_xlabel("1er eigenvector")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2do eigenvector")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3ro eigenvector")
        ax.w_zaxis.set_ticklabels([])
        plt.show()
        st.pyplot(fig)
    with row1_6:
        fig = plt.figure(1, figsize=(10,6))
        plt.scatter(X_reduced[:, 0],  X_reduced[:, 1], c=df2['cat'],cmap=plt.cm.Paired,linewidths=10)
        plt.annotate('Ver el cluster café',xy=(20,-20),xytext=(9,8),arrowprops=dict(facecolor='black', shrink=0.05))
        #plt.scatter(test_reduced[:, 0],  test_reduced[:, 1],c='r')
        plt.title("Este el 2D Transformado ")
        plt.show()
        st.pyplot(fig)

with st.expander("Contáctanos👉"):
    st.subheader('Quieres conocer mas de IA, ML o DL 👉[contactanos!!](http://ia.smartecorganic.com.co/index.php/contact/)')
