from pymoo.core.problem import Problem
import multiprocessing
import TrainEvalNet as ten
import sys
import numpy as np

#n_proccess = 10
#pool = multiprocessing.Pool(n_proccess)

class NetworkOptimization(Problem):
    def __init__(self, seed, cpu_cores=10):
        '''
            n_var : int                     Number of Variables
            n_obj : int                     Number of Objectives
            n_ieq_constr : int              Number of Inequality Constraints
            n_eq_constr : int               Number of Equality Constraints
            xl : np.array, float, int       Lower bounds for the variables. if integer all lower bounds are equal.
            xu : np.array, float, int       Upper bounds for the variable. if integer all upper bounds are equal.
            vtype : type                    The variable type. So far, just used as a type hint.
        '''
        super().__init__(n_var=5, 
                        n_obj=2,
                        xl=np.array([0, 0, 3, 0, 0]), 
                        xu=np.array([1, 1, 50, 100, 100]), 
                        vtype=float)
        
        self.n_process = cpu_cores
        self.pool = multiprocessing.Pool(self.n_process)
        self.seed = seed

    def _evaluate(self, x, out, *args, **kwargs):
        # x1 = NoiseLevel   x[:, 0]
        # x2 =  leaking     x[:, 1]
        # x3 = reservoir    x[:, 2]
        # x4 = seed 1   W    x[:, 3]
        # x5 = seed 2   P   x[:, 4]

        f = np.zeros((x.shape[0], 1))
        f2 = np.zeros((x.shape[0], 1))

        grupos = len(x) // self.n_process
        manager = multiprocessing.Manager()
        return_dict1 = manager.dict()
        return_dict2 = manager.dict()
        jobs = []
        for g  in range(grupos):
            #print("Procesando")
            for k in range(self.n_process):
                i = (self.n_process * g) + k
                x1 = x[i, 0]
                x2 = x[i, 1]
                x3 = int(x[i, 2])
                x4 = int(x[i, 3])
                x5 = int(x[i, 4])
                t1 = multiprocessing.Process(target=self.funcion_multiproceso, args=(x1, x2, x3, x4, x5, 1, 1, k, return_dict1, return_dict2)  )
                jobs.append(t1)
                t1.start()
        #print(jobs)
            for proc in jobs:
                proc.join()

            for k in range(self.n_process):
                i = (self.n_process * g) + k
                f[i,0] = return_dict1[k]
                f2[i,0] = return_dict2[k]
                #print("---"+str(k) + " " +str(f[i, 0])+ " " +str(f2[i, 0])+  " " +str(f3[i,0]))
                name = "MejorEvo/"+str(self.seed)+".txt"
                d=open(name,'a')
                a = np.array([f[i,0], f2[i,0], x[i,0], x[i,1], x[i, 2], x[i, 3], x[i, 4]])
                np.savetxt(d, a, newline=" ")
                d.write("\n")
                d.close()
            
            jobs.clear()

        out["F"] = np.column_stack([f , f2])


    def funcion_multiproceso(self, x1, x2, x3, x4, x5, entradas, salidas, k, return_dict1, return_dict2):
        f1, f2  = ten.regresionESN(x1, x2, x3, x4, x5, entradas, salidas)
    
        return_dict1[k] = f1
        return_dict2[k] = f2 
    
    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict