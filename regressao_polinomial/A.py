import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('regressao_polinomial/boston.csv')

x = data.iloc[:, :-1] #pegando as treze primeiras colunas como atributos 
y = data.iloc[:, -1]  #pegando última coluna como saída

#Divide o conjunto de dados em treino (80%) e teste (20%)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)
