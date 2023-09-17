import pandas as pd
from sklearn.model_selection import train_test_split

# Carregue o conjunto de dados a partir do arquivo CSV
data = pd.read_csv('boston.csv')

# Separe os atributos (13 primeiras colunas) e as saídas (última coluna)
x = data.iloc[:, :-1]  # Atributos
y = data.iloc[:, -1]   # Saídas

# Divida o conjunto de dados em treino (80%) e teste (20%)
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, random_state=42)

# O argumento "test_size" define a proporção do conjunto de teste (20% neste caso)
# O argumento "random_state" é usado para garantir a reprodutibilidade do resultado

# Agora você tem os conjuntos de treino e teste prontos para serem usados em sua análise ou modelo de machine learning.
