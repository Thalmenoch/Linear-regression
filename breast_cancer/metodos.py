import matplotlib.pyplot as plt

# Função para calcular a acurácia usando TP, FP, TN, FN
def calcular_acuracia(tp, fp, tn, fn):
    total = tp + fp + tn + fn
    return (tp + tn) / total

# Função para calcular a revocação (recall) usando TP e FN
def calcular_revocacao(tp, fn):
    if (tp + fn) == 0:
        return 0.0  # Lidar com o caso especial quando TP e FN são zero
    else:
        return tp / (tp + fn)

# Função para calcular a precisão usando TP e FP
def calcular_precisao(tp, fp):
    if (tp + fp) == 0:
        return 0.0  # Lidar com o caso especial quando TP e FP são zero
    else:
        return tp / (tp + fp)

# Função para calcular o F1-Score usando TP e FN
def calcular_f1_score(tp, fn):
    if tp == 0 and fn == 0:
        return 0.0  # Lidar com o caso especial quando TP e FN são zero
    else:
        precisao = calcular_precisao(tp, 0)  # Nenhum FP
        revocacao = calcular_revocacao(tp, fn)
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