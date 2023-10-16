import numpy as np
import matplotlib.pyplot as plt

def calcular_acuracia(y_verdadeiro, y_predito):
    return np.mean(y_verdadeiro == y_predito)

# Função para calcular a revocação
def calcular_revocacao(y_verdadeiro, y_predito):
    tp = np.sum((y_verdadeiro == 1) & (y_predito == 1))
    fn = np.sum((y_verdadeiro == 1) & (y_predito == 0))
    if (tp + fn) == 0:
        return 0.0  # Lidar com o caso especial quando TP e FN são zero
    else:
        return tp / (tp + fn)

# Função para calcular a precisão
def calcular_precisao(y_verdadeiro, y_predito):
    tp = np.sum((y_verdadeiro == 1) & (y_predito == 1))
    fp = np.sum((y_verdadeiro == 0) & (y_predito == 1))
    if (tp + fp) == 0:
        return 0.0  # Lidar com o caso especial quando TP e FP são zero
    else:
        return tp / (tp + fp)

# Função para calcular o F1-Score
def calcular_f1_score(y_verdadeiro, y_predito):
    precisao = calcular_precisao(y_verdadeiro, y_predito)
    revocacao = calcular_revocacao(y_verdadeiro, y_predito)
    
    if precisao == 0 or revocacao == 0:
        return 0.0  # Lidar com o caso especial quando ambos precisao e revocacao são zero
    else:
        return 2 * (precisao * revocacao) / (precisao + revocacao)

def calcular_matriz_confusao(y_real, y_predito):
    TP = FP = TN = FN = 0
    for y_real_i, y_predito_i in zip(y_real, y_predito):
        if y_real_i == 1 and y_predito_i == 1:
            TP += 1
        elif y_real_i == 0 and y_predito_i == 1:
            FP += 1
        elif y_real_i == 0 and y_predito_i == 0:
            TN += 1
        elif y_real_i == 1 and y_predito_i == 0:
            FN += 1
    return TP, FP, TN, FN

def plot_confusion_matrix(TP, FP, TN, FN):
    confusion_matrix = [[TP, FP], [FN, TN]]

    plt.figure(figsize=(6, 6))
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title('Matriz de Confusão')
    plt.colorbar()

    tick_marks = ['Positivos', 'Negativos']
    plt.xticks([0, 1], tick_marks, rotation=45)
    plt.yticks([0, 1], tick_marks)

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(confusion_matrix[i][j]), ha='center', va='center', color='black', fontsize=16)

    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.show()

def plot_metrics(acuracia, precisao, revocacao, f1):
    labels = ['Acurácia', 'Precisão', 'Revocação', 'F1-Score']
    valores = [acuracia, precisao, revocacao, f1]

    plt.bar(labels, valores, color=['blue', 'green', 'red', 'purple'])
    plt.ylim(0, 1.1)  # Defina os limites do eixo y de 0 a 1.1 para as métricas

    for i, v in enumerate(valores):
        plt.text(i, v + 0.03, f'{v:.2f}', ha='center', va='center', fontsize=12, fontweight='bold', color='black')

    plt.title('Métricas de Desempenho')
    plt.ylabel('Valor')
    plt.show()