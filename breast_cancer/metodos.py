def calcular_acuracia(y_verdadeiro, y_predito):
    return np.mean(y_verdadeiro == y_predito)

# Função para calcular a revocação
def calcular_revocacao(y_verdadeiro, y_predito):
    tp = np.sum((y_verdadeiro == 1) & (y_predito == 1))
    fn = np.sum((y_verdadeiro == 1) & (y_predito == 0))
    return tp / (tp + fn)

# Função para calcular a precisão
def calcular_precisao(y_verdadeiro, y_predito):
    tp = np.sum((y_verdadeiro == 1) & (y_predito == 1))
    fp = np.sum((y_verdadeiro == 0) & (y_predito == 1))
    return tp / (tp + fp)

# Função para calcular o F1-Score
def calcular_f1_score(y_verdadeiro, y_predito):
    precisao = calcular_precisao(y_verdadeiro, y_predito)
    revocacao = calcular_revocacao(y_verdadeiro, y_predito)
    return 2 * (precisao * revocacao) / (precisao + revocacao)