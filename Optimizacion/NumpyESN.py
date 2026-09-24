import BackNumpy as B
import math
import numpy as np

'''
            noise               [0,1]
            leaking             [0,1]
            reservoir           [5,1000]
            weights_seed        [0,100]
            permutation_seed    [0,100]
            density             [0.1,1]
'''
class PredictionESN:
    def __init__(self, noise : float, 
                        leaking : float,
                        reservoir : int,
                        weights_seed : int,
                        permutation_seed : int,  
                        inputs : int, 
                        outputs : int, 
                        density : float = 0.1,
                        bias = 1.0, 
                        function = 'tanh',
                        fit_method = 'inc', 
                        noise_bias = None,
                        save : bool = False, 
                        Win = None, 
                        Wres = None, 
                        Wout = None):
        
        if noise >= 0.0 and noise <= 1.0:
            self.ruido = noise
        else:
            raise ValueError("Noise value must be between 0 and 1.")

        if leaking >= 0.0 and leaking <= 1.0:
            self.fuga = leaking
        else:
            raise ValueError("Leaking rate value must be between 0 and 1.")

        if density >= 0.1 and density <= 1.0:
            self.densidad = density
        else:
            raise ValueError("Density value must be between 0.1 and 1.")

        if reservoir >= 1:
            self.contenedor = reservoir
        else:
            raise ValueError("The size of the reservoir must be greater than 0.")

        if inputs >= 1:
            self.entrada = inputs
        else:
            raise ValueError("The number of imputs must be greater than 0.")
        
        if outputs >= 1:
            self.salidas = outputs
        else:
            raise ValueError("The number of outputs must be greater than 0.")

        if fit_method == 'inc' or fit_method == 'standard':
            if fit_method == 'inc':
                self.fit_method = 'inc'
            if fit_method == 'standard':
                self.fit_method = 'standard'
        else:
            raise ValueError("Fit method can only be inc or standard.")

        if function == 'tanh' or function == 'leaky_relu' or function == 'relu':
            if function == 'tanh':
                self.activation = B.TanH
            if function == 'relu':
                self.activation = B.activation_relu
            if function == 'leaky_relu':
                self.activation = B.activation_leaky_relu
        else:
            raise ValueError("Activation function can only be tanh or relu.")

        if Wout is not None:
            self.Wout = Wout
        
        if noise_bias is not None:
            self.noise_bias = noise_bias

        self.semillaW = weights_seed
        self.semillaP = permutation_seed
        
        self.bias = bias
        self.outputBias=1.0,
        self.outputInputScaling=1.0
        self.feedbackScaling=1.0

        self.save = save

        self.out_inverse_activation = lambda x: x
        self.out_activation = lambda x: x
        
        self.matriz_entrada(Win)
        self.matriz_contenedor(Wres)



    '''
        Use this function if the amount of data is not too big
    '''
    def fit(self, DatosX, DatosY):

        if(DatosX.shape[0] != DatosY.shape[0]):
            raise ValueError("Las dimensiones de los datos de entrada y salida no coinciden.")

        muestras = DatosX.shape[0]

        self._X = B.Empty((1 + self.entrada + self.contenedor, muestras))
        self.xmin = B.Zeros((self.contenedor, 1))
        Y_target = B.Empty((self.salidas, muestras))
       
     
        for i in range(muestras):
            u = DatosX[i].reshape(self.entrada, 1)
            T = B.Dot(self.Win, B.Vs((B.Array(self.bias), u))) + B.Dot(self.W, self.xmin)
            self.xmin = self.xmin * (1 - self.fuga)
            self.xmin += self.fuga * self.activation(T + self.noise_bias)
            self._X[:,i] = B.Vs( (B.Array(self.outputBias), self.outputInputScaling * u, self.xmin) )[:, 0]
            Y_target[:,i] = self.out_inverse_activation(DatosY[i]).T

        self.Wout = B.Dot(Y_target, B.Pinv(self._X))
        
        if self.save == True:
            save_matrices(True)


    '''
        Use this function if the amount of data is too big
    '''

    def incremental_fit(self, DatosX, DatosY, incializar_estado=True):
        if(DatosX.shape[0] != DatosY.shape[0]):
            raise ValueError("Las dimensiones de los datos de entrada y salida no coinciden.")


        #n es el número de muestras y m es el número de entradas
        n, e = DatosX.shape
        #print(n)
        #print(m)

        muestras = DatosX.shape[0]
        #print(muestras)
        over = 0
        #En caso de que ya se haya inicializado, no se vuelve a resetear el estado anterior
        if incializar_estado:
            self.xmin = B.Zeros((self.contenedor, 1))
        

        # es el valor del estado con el sesgo y el estado eco
        m = 1 + self.entrada + self.contenedor

        #print(m)
        #print("contenedor")
        #print(self.contenedor)
        self._X = B.Empty((m, m))
        
        #print(self._X.shape)

        # --------------------------- Inicialización ----------------------------------------------------
        # Se va crear una matriz de inicialización de m x m 
        

        b = DatosY
        if incializar_estado:
            # ------------------ Transmisión de la señal en la red --------------------------------------
            for i in range(m):
                u = DatosX[i].reshape(self.entrada, 1)
                #print(u.shape)
                T = B.Dot(self.Win, B.Vs((B.Array(self.bias), u))) + B.Dot(self.W, self.xmin)
                self.xmin = self.xmin * (1 - self.fuga)
                self.xmin += self.fuga * self.activation(T + self.noise_bias)
                self._X[:,i] = B.Vs( (B.Array(self.outputBias), self.outputInputScaling * u, self.xmin) )[:, 0]
            # ------------------ Transmisión de la señal en la red --------------------------------------
            
            #------------------ Cálculo de la matriz de salida -------------------------------------------
            # b = n x o
            b = DatosY
            # C = m x m
            C = np.transpose(self._X)
            # H = m x m
            self.H = B.Pinv( np.transpose(C) @ C )
            #print(self.H.shape)
            # X = (m x m) ((m x m)(m x o) )
            # X = (m x m) ( m x o )
            # X = (m x o)
            self.x = self.H @ ( np.transpose(C) @ b[0:m])
            #------------------ Cálculo de la matriz de salida -------------------------------------------
            i = m
        else:
            i = 0    
        # ------------------------------ Inicialización --------------------------------------------------
        
        # ---------------- Transmisión y cálculo de la matriz de salida ----------------------------------
        while i < muestras:
            # ------------------ Transmisión de la señal en la red --------------------------------------
            u = DatosX[i].reshape(self.entrada, 1)
            T = B.Dot(self.Win, B.Vs((B.Array(self.bias), u))) + B.Dot(self.W, self.xmin)
            self.xmin = self.xmin * (1 - self.fuga)
            self.xmin += self.fuga * self.activation(T + self.noise_bias)
            c = B.Vs( (B.Array(self.outputBias), self.outputInputScaling * u, self.xmin) )
            #------------------ Transmisión de la señal en la red ---------------------------------------

            #------------------ Cálculo de la matriz de salida -------------------------------------------

            # s = (m x m) (m x 1) = (m x 1)
            s = self.H @ c
            # H = (m x m) - ((m x 1) (1 x m) ) / (1 + (1 x m)(m x 1) )
            # H = (m x m) - (m x m) ) / (1 + (1) )
            # H = (m x m) - (m x m) 
            # H = (m x m)
            self.H = self.H - (s @ s.T)/(1.0 + c.T @ s )
            # v = (m x m)(m x 1) 
            # v = (m x 1) 
            v = self.H @ c
            # x = (m x o) + (m x 1) ((1 x o) - (1 x m) (m x o) )
            # x = (m x o) + (m x 1) ((1 x o) - (1 x o))
            # x = (m x o) + (m x 1) ((1 x o))
            #self.x = self.x + v*( b[i].reshape((1,1)) - c.T @ self.x )
            self.x = self.x + v @ ( b[i] - c.T @ self.x )
        
            #------------------ Cálculo de la matriz de salida -------------------------------------------
            tmp = np.sum(self.x)
            if (np.isnan(tmp) or np.isinf(tmp))  :
                #print("NAN encontrado")
                over = 1
                break
            i += 1

        # ---------------- Transmisión y cálculo de la matriz de salida ----------------------------------
        if over == 1:
            self.Wout = 1
        else:
            self.Wout = np.transpose(self.x)
        
        if self.save == True:
            save_matrices(True)

    def predictCompacta(self, DatosX, continuacion=False):
        muestras = DatosX.shape[0]
    
        if not continuacion:
            self.xmin = B.Zeros((self.contenedor, 1))

        #Se debe de guardar también el vector de retroalimentación
        self._X = B.Empty((1 + self.entrada + self.contenedor, muestras))
        for i in range(muestras):
            u = DatosX[i].reshape(self.entrada, 1)
            T = B.Dot(self.Win, B.Vs((B.Array(self.bias), u))) + B.Dot(self.W, self.xmin)
            self.xmin = self.xmin * (1 - self.fuga)
            self.xmin += self.fuga * self.activation(T + self.noise_bias)
            self._X[:,i] = B.Vs( (B.Array(self.outputBias), self.outputInputScaling * u, self.xmin) )[:, 0]
        

        Y = B.Dot(self.Wout, self._X)

        return Y.T

    def predictRetroCompacta(self, DatosX, estado_eco, continuacion=False):
        muestras = DatosX.shape[0]
    
        if not continuacion:
            eco_state = B.Zeros((self.contenedor, 1))
        else: 
            eco_state = estado_eco

        #Se debe de guardar también el vector de retroalimentación
        
        #u = DatosX[i].reshape(self.entrada, 1)
        u = DatosX.reshape(self.entrada, 1)
        #print(u.shape)
        T = B.Dot(self.Win, B.Vs((B.Array(self.bias), u))) + B.Dot(self.W, eco_state)
        eco_state = eco_state * (1 - self.fuga)
        eco_state += self.fuga * self.activation(T + self.noise_bias)
        X = B.Vs( (B.Array(self.outputBias), self.outputInputScaling * u, eco_state) )[:, 0]
        Y = B.Dot(self.Wout, X)
        #print(Y)
        return Y, eco_state
     
    def matriz_entrada(self, Win=None):
        if Win is None:
            B.Seed(self.semillaW)
            self.noise_bias = (B.Random() - 0.5) * self.ruido
            self.Win = B.Random(self.contenedor, 1 + self.entrada) - 0.5
        else:
            self.Win = Win
        
    def matriz_contenedor(self, Wres=None):
        if Wres is None:
            self.permutacion()
            self.calculardensidad()
            Grupos = math.ceil(self.celdas / self.contenedor)
            B.Seed(self.semillaW)
            diagonal=B.Random(self.contenedor, Grupos)-0.5

            self.W = B.Zeros((self.contenedor, self.contenedor))
            #self.Ws = B.Zeros((self.contenedor, 2), dtype=int)
            for i in range(self.contenedor):
                for j in range(Grupos):
                    encaje = self.p[i] + j
                    if encaje > (self.contenedor - 1):
                        encaje = encaje - self.contenedor
            
                    self.W[i,encaje] = diagonal[self.p[i], j]
                    #self.Ws[i,0] = i
                    #self.Ws[i,1] = encaje

            #print(np.linalg.eig(self.W)[0])
            Weigen =B.Abs( B.Eigen(self.W)[0] )
            #print(Weigen)
            self.W *= 1 / B.Max(Weigen)
        else:
            self.W = Wres
        '''
        for i in range(self.contenedor):
            self.Ws[i,0] = self.W[int(self.Ws[i,1]), int(self.Ws[i,2])]
        '''

    def permutacion(self):
        B.Seed(self.semillaP)
        self.p = B.Permutation(self.contenedor)
    
    def calculardensidad(self):
        # Se calcula el número de celdas que se van hacer diferentes de 0
        # f(densidad) = celdas
        # celdas = A*densidad + B*densidad + C
        n2 = self.contenedor * self.contenedor
        n = self.contenedor
        a = -5/18*n2 + 25/9*n
        b = 17/12*n2 - 25/6*n
        c = -5/36*n2 + 25/18*n
        self.celdas = math.floor( a*(self.densidad*self.densidad) + b*self.densidad + c)

    def save_matrices(self, save=False):
        np.savetxt("Wout.txt", self.Wout)
        np.savetxt("Win.txt", self.Win)
        np.savetxt("Wres.txt", self.W)
        x = np.zeros((7))
        x[0] = self.ruido
        x[1] = self.fuga
        x[2] = self.contenedor // 1
        x[3] = self.semillaW // 1
        x[4] = self.semillaP // 1
        x[5] = self.noise_bias
        x[6] = 1.0 - self.fuga
        np.savetxt("Encoding.txt", x)
        np.savetxt("P.txt", self.p, fmt = '%d')

class ClasificacionESN:
    def __init__(self, ruido, fuga, contenedor, semillaW, semillaP, densidad, bias, entrada, clases, funcion):
        '''
            noise       [0,1]
            leaking rate  [0,1]
            contenedor  [5,1000]
            semillaW    [0,100]
            semillaP    [0,100]
            densidad    [0.1,1]
        '''
        self.ruido = ruido
        self.fuga = fuga
        self.contenedor = contenedor
        self.semillaW = semillaW
        self.semillaP = semillaP
        self.densidad = densidad
        self.bias = bias
        self.funcion = funcion
        self.entrada = entrada
        self.clases = clases
        self.outputBias=1.0,
        self.outputInputScaling=1.0
        self.activation = B.TanH
        self.activation_relu = B.activation_r
        self.out_inverse_activation = lambda x: B.Log((x * 0.98 + 0.01) / (0.99 - x * 0.98))
        self.out_inverse_activation_relu =  B.inv_activation_r
        self.out_activation = lambda x: 0.1 + 0.98 * x / (1 + B.Exp(-x))
        
        self.matriz_entrada()
        self.matriz_contenedor()

    def fitCompacta(self, DatosX, DatosY):
        if(DatosX.shape[0] != DatosY.shape[0]):
            raise ValueError("Las dimensiones de los datos de entrada y salida no coinciden.")
        
        muestras = DatosX.shape[0]
        caracteristicas = DatosX.shape[1]

        if(DatosY.shape[1] > 1 and DatosY.shape[1] != self.clases):
            raise ValueError("El número de clases en los datos de salida no coinciden con el número de clases.")
        
        self.xmin = B.Zeros((self.contenedor, 1))

        self.Xmay = B.Zeros((
            1+self.entrada+self.contenedor, 
            (muestras*caracteristicas)
        ))

        Y_objetivo = B.Zeros((
            self.clases,(muestras*caracteristicas)
        ))

        for i in range(muestras):
            self.xmin = B.Zeros((self.contenedor, 1))
            for j in range(caracteristicas):
                u = DatosX[i,j]
                T = B.Dot(self.Win, B.Vs((B.Array(self.bias), u))) + B.Dot(self.W, self.xmin)
                self.xmin = self.xmin * (1 - self.fuga)
                B.Seed(self.semillaW)
                if(self.funcion == 1):
                    self.xmin += self.fuga * self.activation(T + (B.Random() - 0.5) * self.ruido)
                if(self.funcion == 2):
                    self.xmin += self.fuga * self.activation_relu(T + (B.Random() - 0.5) * self.ruido)
                
                self.Xmay[:,i*caracteristicas+j] = B.Vs( (B.Array(self.outputBias), self.outputInputScaling * u, self.xmin)  )[:, 0]
                
                if(self.funcion == 1):
                    Y_objetivo[:,i*caracteristicas+j] = self.out_inverse_activation(DatosY[i]).T
                if(self.funcion == 2):
                    Y_objetivo[:,i*caracteristicas+j] = self.out_inverse_activation_relu(DatosY[i]).T
                
        self.Wout = B.Dot(Y_objetivo, B.Pinv(self.Xmay))

    def predictCompacta(self, DatosX):
        caracteristicas = DatosX.shape[1]
        muestras = DatosX.shape[0]
        
        Y = B.Empty((DatosX.shape[0], self.clases))
        self.Xmay = B.Zeros((
            1+self.entrada+self.contenedor, caracteristicas
        ))

        for i in range(muestras):
            self.xmin = B.Zeros((self.contenedor, 1))
            for j in range(caracteristicas):
                u = DatosX[i,j]
                T = B.Dot(self.Win, B.Vs((B.Array(self.bias), u))) + B.Dot(self.W, self.xmin)
                self.xmin = self.xmin * (1 - self.fuga)
                B.Seed(self.semillaW)
                
                if(self.funcion == 1):
                    self.xmin += self.fuga * self.activation(T + (B.Random() - 0.5) * self.ruido)
                if(self.funcion == 2):
                    self.xmin += self.fuga * self.activation_relu(T + (B.Random() - 0.5) * self.ruido)
                
                self.Xmay[:,j] = B.Vs( (B.Array(self.outputBias), self.outputInputScaling * u, self.xmin)  )[:, 0]
            
            y = B.Dot(self.Wout, self.Xmay)
            
            Y[i] = B.Mean(y,1)
        
        resultado = B.Zeros(Y.shape)
        
        for i, n in enumerate(B.ArgMax(Y,1)):
            resultado[i, n] = 1.0
   
        return resultado

    def mult_dispersa(self, x):
        R = B.Zeros((x.shape))
        for i in range(len(self.W)):
            R[i] = self.W[i,self.Ws[i,1]] * x[self.Ws[i,1],0]
        return R
     

    def matriz_entrada(self):
        B.Seed(self.semillaW) 
        self.Win = B.Random(self.contenedor, 2) - 0.5
        
    def matriz_contenedor(self):
        self.permutacion()
        self.calculardensidad()
        Grupos = math.ceil(self.celdas / self.contenedor)
        B.Seed(self.semillaW)
        diagonal=B.Random(self.contenedor, Grupos)-0.5

        self.W = B.Zeros((self.contenedor, self.contenedor))
        #self.Ws = B.Zeros((self.contenedor, 2), dtype=int)
        for i in range(self.contenedor):
            for j in range(Grupos):
                encaje = self.p[i] + j
                if encaje > (self.contenedor - 1):
                    encaje = encaje - self.contenedor
        
                self.W[i,encaje] = diagonal[self.p[i], j]
                #self.Ws[i,0] = i
                #self.Ws[i,1] = encaje

        #print(np.linalg.eig(self.W)[0])
        Weigen =B.Abs( B.Eigen(self.W)[0] )
        #print(Weigen)
        self.W *= 1 / B.Max(Weigen)
        '''
        for i in range(self.contenedor):
            self.Ws[i,0] = self.W[int(self.Ws[i,1]), int(self.Ws[i,2])]
        '''


    def permutacion(self):
        B.Seed(self.semillaP)
        self.p = B.Permutation(self.contenedor)
    
    def calculardensidad(self):
        # Se calcula el número de celdas que se van hacer diferentes de 0
        # f(densidad) = celdas
        # celdas = A*densidad + B*densidad + C
        n2 = self.contenedor * self.contenedor
        n = self.contenedor
        a = -5/18*n2 + 25/9*n
        b = 17/12*n2 - 25/6*n
        c = -5/36*n2 + 25/18*n
        self.celdas = math.floor( a*(self.densidad*self.densidad) + b*self.densidad + c)
