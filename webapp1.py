import streamlit as st
import pandas as pd
import telnetlib
import subprocess
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

def user_input_features():
    sepal_length = st.sidebar.slider('Длина чашелистика', 4.0, 9.9, 3.5)
    sepal_width = st.sidebar.slider('Ширина чашелистика', 2.0, 4.5, 2.6)
    petal_length = st.sidebar.slider('Длина лепестка', 1.0, 7.0, 2.2)
    petal_width = st.sidebar.slider('Ширина лепестка', 0.1, 3.0, 0.7)
    data = {'Длина чашелистика': sepal_length,
            'Ширина чашелистика': sepal_width,
            'Длина лепестка': petal_length,
            'Ширина лепестка': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

iris = datasets.load_iris()
X = iris.data
Y = iris.target

st.write("""
# Прогнозирование вида цветка ириса
""")

st.sidebar.header('Параметры пользователя')
df = user_input_features()

st.subheader('Параметры пользователя')
st.write(df)

clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Номер вида Ириса')
st.write(iris.target_names)

st.subheader('Вероятность прогнозирования')
st.write(prediction_proba)

st.subheader('Прогноз')
st.write(iris.target_names[prediction])
