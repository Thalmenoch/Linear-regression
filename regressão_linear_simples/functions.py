import numpy as np

#Cálculo do MSE
def MSE(y, y_previsto):
    return np.mean((y - y_previsto) ** 2)

#Cálculo dos coeficientes
def coef_linear_angular(x, y, media_x, media_y):
    coef_angular = np.sum((x - media_x) * (y - media_y)) / np.sum((x - media_x) ** 2)
    coef_linear = media_y - coef_angular * media_x

    return coef_angular, coef_linear 