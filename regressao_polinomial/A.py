import pandas as pd

data = pd.read_csv('regressao_polinomial/boston.csv')

# Suponha que você tenha todas as colunas no arquivo 'boston.csv', incluindo a coluna 'target' como o alvo
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Dividir os dados em conjuntos de treinamento e teste (80% treinamento, 20% teste)
split_ratio = 0.8
split_index = int(len(X) * split_ratio)

X_train, X_test, y_train, y_test = X[:split_index], X[split_index:], y[:split_index], y[split_index:]
