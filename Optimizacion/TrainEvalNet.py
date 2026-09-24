from NumpyESN import PredictionESN
from sklearn.metrics import mean_squared_error
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
import numpy as np


def regresionESN(x1, x2, x3, x4, x5, inputs, outputs):
    
    X_entrena = np.loadtxt("../Dataset/Preprocess/Xtrain.txt")
    Y_entrena = np.loadtxt("../Dataset/Preprocess/Ytrain.txt")

    X_entrena = X_entrena.reshape((X_entrena.shape[0], 1))
    Y_entrena = Y_entrena.reshape((Y_entrena.shape[0], 1))

    X_prueba = np.loadtxt("../Dataset/Preprocess/Xtest.txt")
    Y_prueba = np.loadtxt("../Dataset/Preprocess/Ytest.txt")

    X_prueba = X_prueba.reshape((X_prueba.shape[0], 1))
    Y_prueba = Y_prueba.reshape((Y_prueba.shape[0], 1))

    esn = PredictionESN(x1, x2, x3, x4, x5, inputs, outputs, function='relu')
    
    esn.incremental_fit(X_entrena, Y_entrena, True)
    y_pred = esn.predictCompacta(X_prueba)

    RMSE = root_mean_squared_error(Y_prueba, y_pred)
    return RMSE, x3
