from NumpyESNClasif import ESNClasificador
from sklearn.metrics import accuracy_score
import numpy as np


def entrenar_modelo(x1, x2, x3, x4, x5, topology='full'):
    # x1 = spectral_radius
    # x2 = sparsity          (solo aplica si topology == 'full')
    # x3 = reservoir_size
    # x4 = seed_in    (Win / bias)
    # x5 = seed_res   (Wres)

    X_entrena = np.loadtxt("Dataset/Preprocess/Xtrain.txt")
    Y_entrena = np.loadtxt("Dataset/Preprocess/Ytrain.txt")

    X_prueba = np.loadtxt("Dataset/Preprocess/Xtest.txt")
    Y_prueba = np.loadtxt("Dataset/Preprocess/Ytest.txt")

    Y_entrena = Y_entrena.reshape(-1, 1)
    Y_prueba = Y_prueba.reshape(-1, 1)

    esn = ESNClasificador(
        inputSize=X_entrena.shape[1],
        reservoir_size=x3,
        seed_in=x4,
        seed_res=x5,
        spectral_radius=x1,
        sparsity=x2,
        topology=topology,
    )

    try:
        esn.train(X_entrena, Y_entrena)

        y_pred = esn.predict(X_prueba)
        y_pred = (y_pred > 0.5).astype(int)

        accuracy = accuracy_score(Y_prueba, y_pred)
    except (ValueError, np.linalg.LinAlgError) as e:
        # Con reservorios muy pequenos, ReLU puede dejar neuronas "muertas"
        # (columna de ceros) para todos los ejemplos, lo que hace que la
        # matriz usada en el entrenamiento (MGS) sea rank-deficient. En vez
        # de tirar la busqueda evolutiva, se penaliza el individuo.
        esn.W_out = np.zeros((esn.reservoir_size, Y_entrena.shape[1]))
        accuracy = 0.0
        print(f"topologia={topology}  reservorio={x3:4d}  spectral_radius={x1:.4f}  sparsity={x2:.4f}  "
              f"seed_in={x4:3d}  seed_res={x5:3d}  FALLO ({e}) -> accuracy=0.0000", flush=True)
        return esn, accuracy

    print(f"topologia={topology}  reservorio={x3:4d}  spectral_radius={x1:.4f}  sparsity={x2:.4f}  "
          f"seed_in={x4:3d}  seed_res={x5:3d}  accuracy={accuracy:.4f}", flush=True)

    return esn, accuracy


def clasificacionESN(x1, x2, x3, x4, x5, topology='full'):
    esn, accuracy = entrenar_modelo(x1, x2, x3, x4, x5, topology=topology)
    return 1.0 - accuracy, x3
