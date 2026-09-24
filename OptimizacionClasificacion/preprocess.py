import os
import numpy as np
import scipy.io

'''
    Genera un split train/test fijo del dataset Lorenz3D usado en
    Original/Hybrid ESN_Classif_MGS.ipynb, para que todas las evaluaciones
    de la busqueda evolutiva usen exactamente los mismos datos.

    N: factor de decimacion de features (mismo procedimiento que la celda de
       downsampling del notebook: data[:, ::N]).

    Las series tienen 3000 pasos. Se busca dejar la red con un maximo de
    15 a 20 entradas, por lo que N debe estar entre 150 (20 features) y
    200 (15 features). Se usa el extremo superior del rango (20 features)
    para retener la mayor cantidad de informacion posible.
'''

DECIMATION_N = 150
TEST_SIZE = 0.2
SPLIT_SEED = 0

ORIGINAL_DIR = os.path.join("..", "Original")
OUT_DIR = os.path.join("Dataset", "Preprocess")


def main():
    from sklearn.model_selection import train_test_split

    mat_data = scipy.io.loadmat(os.path.join(ORIGINAL_DIR, "Lorenz3D.mat"))
    data = mat_data["Lorenz3D"]

    mat_labels = scipy.io.loadmat(os.path.join(ORIGINAL_DIR, "LE_lorenz.mat"))
    labels = mat_labels["LE_lorenz"]

    data_dec = data[:, ::DECIMATION_N]

    X_train, X_test, y_train, y_test = train_test_split(
        data_dec, labels, test_size=TEST_SIZE, random_state=SPLIT_SEED
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    np.savetxt(os.path.join(OUT_DIR, "Xtrain.txt"), X_train)
    np.savetxt(os.path.join(OUT_DIR, "Ytrain.txt"), y_train)
    np.savetxt(os.path.join(OUT_DIR, "Xtest.txt"), X_test)
    np.savetxt(os.path.join(OUT_DIR, "Ytest.txt"), y_test)

    print(f"N={DECIMATION_N} -> features={data_dec.shape[1]}")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")


if __name__ == "__main__":
    main()
