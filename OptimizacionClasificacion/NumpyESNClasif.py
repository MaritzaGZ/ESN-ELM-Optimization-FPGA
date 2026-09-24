import os
import numpy as np
import BackNumpy as B

'''
    Puerto de la clase ESN del notebook Original/Hybrid ESN_Classif_MGS.ipynb
    a una version seedeada, para poder usarla dentro de la busqueda evolutiva
    (el notebook original no fijaba semillas, lo que impedia reproducir la
    evaluacion de un mismo individuo entre corridas).

    La activacion del notebook original (tanh) se reemplazo por ReLU, de
    menor costo computacional.

    spectral_radius     [0.1, 1.5]
    sparsity            [0.0, 1.0]
    reservoir_size      [2, 50]
    seed_in             semilla para W_in y bias
    seed_res            semilla para W_res
'''
class ESNClasificador:
    def __init__(self, inputSize, reservoir_size, seed_in, seed_res,
                 spectral_radius=0.9, sparsity=0.1, topology='full'):
        self.inputSize = inputSize
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.topology = topology
        self.seed_in = seed_in
        self.seed_res = seed_res

        B.Seed(seed_in)
        self.W_in = np.random.uniform(-0.5, 0.5, (self.reservoir_size, self.inputSize))
        self.bias = np.random.uniform(0, 1, (1, self.reservoir_size))

        B.Seed(seed_res)
        if topology == 'full':
            self.W_res = self._initialize_full(sparsity)
        elif topology == 'ring':
            self.W_res = self._initialize_ring()
        else:
            raise ValueError("Invalid topology. Choose 'full' or 'ring'.")

        self._scale_spectral_radius()

        self.W_out = None

    def _initialize_full(self, sparsity):
        W_res = np.random.uniform(-0.5, 0.5, (self.reservoir_size, self.reservoir_size))
        mask = np.random.rand(*W_res.shape) > sparsity
        W_res[mask] = 0.0
        return W_res

    def _initialize_ring(self):
        W_res = np.zeros((self.reservoir_size, self.reservoir_size))
        for i in range(self.reservoir_size - 1):
            W_res[i + 1, i] = np.random.uniform(-0.5, 0.5)
        W_res[0, self.reservoir_size - 1] = np.random.uniform(-0.5, 0.5)
        return W_res

    def _scale_spectral_radius(self):
        eigenvalues = np.linalg.eigvals(self.W_res)
        max_eigenval = np.max(np.abs(eigenvalues))
        if max_eigenval > 0:
            self.W_res = self.W_res * (self.spectral_radius / max_eigenval)

    def run_reservoir(self, X):
        reservoir_states = np.zeros((len(X), self.reservoir_size))
        for t in range(1, len(X)):
            reservoir_states[t, :] = B.activation_relu(
                np.dot(self.W_res, reservoir_states[t - 1, :]) + np.dot(self.W_in, X[t])
            )
        return reservoir_states

    def train(self, X, y):
        H = B.activation_relu(np.dot(X, self.W_in.T) + self.bias)

        m, n = H.shape
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
        V = H.astype(np.float64).copy()

        for i in range(n):
            R[i, i] = np.linalg.norm(V[:, i])
            if np.isclose(R[i, i], 0):
                raise ValueError("Matrix is rank deficient.")

            Q[:, i] = V[:, i] / R[i, i]

        # NOTA: en el notebook original este bucle queda fuera del "for i"
        # (se ejecuta una sola vez, con el ultimo valor de i). Se conserva
        # tal cual para que el modelo optimizado sea exactamente el mismo
        # que el validado en Original/Hybrid ESN_Classif_MGS.ipynb.
        for j in range(i + 1, n):
            R[i, j] = np.dot(Q[:, i], V[:, j])
            V[:, j] = V[:, j] - R[i, j] * Q[:, i]

        self.W_out = np.dot(np.dot(np.linalg.pinv(R), Q.T), y)

    def predict(self, X):
        reservoir_states = self.run_reservoir(X)
        y = np.dot(reservoir_states, self.W_out)
        return y

    def guardar_modelo(self, carpeta, accuracy=None):
        '''
            Guarda todas las matrices generadas (Win, bias, Wres, Wout) y
            todos los parametros del clasificador (incluyendo el numero de
            neuronas del reservorio) en la carpeta indicada.
        '''
        os.makedirs(carpeta, exist_ok=True)

        np.savetxt(os.path.join(carpeta, "Win.txt"), self.W_in)
        np.savetxt(os.path.join(carpeta, "bias.txt"), self.bias)
        np.savetxt(os.path.join(carpeta, "Wres.txt"), self.W_res)
        np.savetxt(os.path.join(carpeta, "Wout.txt"), self.W_out)

        with open(os.path.join(carpeta, "parametros.txt"), "w") as f:
            f.write(f"reservoir_size={self.reservoir_size}\n")
            f.write(f"inputSize={self.inputSize}\n")
            f.write(f"topology={self.topology}\n")
            f.write(f"spectral_radius={self.spectral_radius}\n")
            f.write(f"sparsity={self.sparsity}\n")
            f.write(f"seed_in={self.seed_in}\n")
            f.write(f"seed_res={self.seed_res}\n")
            if accuracy is not None:
                f.write(f"accuracy={accuracy}\n")
