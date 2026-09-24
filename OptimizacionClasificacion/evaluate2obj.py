from pymoo.core.problem import Problem
import multiprocessing
import TrainEvalNet as ten
import numpy as np


class NetworkOptimization(Problem):
    def __init__(self, seed, topology='full', cpu_cores=10):
        '''
            n_var : int                     Number of Variables
            n_obj : int                     Number of Objectives
            xl : np.array, float, int       Lower bounds for the variables.
            xu : np.array, float, int       Upper bounds for the variables.
            vtype : type                    The variable type. Just used as a type hint.

            topology : str                  'full' or 'ring'. La sparsity (x2)
                                             solo tiene efecto con 'full'.
        '''
        if topology not in ('full', 'ring'):
            raise ValueError("topology debe ser 'full' o 'ring'")

        super().__init__(n_var=5,
                        n_obj=2,
                        xl=np.array([0.1, 0.0, 2, 0, 0]),
                        xu=np.array([1.5, 1.0, 50, 100, 100]),
                        vtype=float)

        self.n_process = cpu_cores
        self.pool = multiprocessing.Pool(self.n_process)
        self.seed = seed
        self.topology = topology

    def _evaluate(self, x, out, *args, **kwargs):
        # x1 = spectral_radius   x[:, 0]
        # x2 = sparsity          x[:, 1]
        # x3 = reservoir_size    x[:, 2]
        # x4 = seed_in           x[:, 3]
        # x5 = seed_res          x[:, 4]

        f = np.zeros((x.shape[0], 1))
        f2 = np.zeros((x.shape[0], 1))

        grupos = len(x) // self.n_process
        manager = multiprocessing.Manager()
        return_dict1 = manager.dict()
        return_dict2 = manager.dict()
        jobs = []
        for g in range(grupos):
            for k in range(self.n_process):
                i = (self.n_process * g) + k
                x1 = x[i, 0]
                x2 = x[i, 1]
                x3 = int(x[i, 2])
                x4 = int(x[i, 3])
                x5 = int(x[i, 4])
                t1 = multiprocessing.Process(target=self.funcion_multiproceso, args=(x1, x2, x3, x4, x5, self.topology, k, return_dict1, return_dict2))
                jobs.append(t1)
                t1.start()

            for proc in jobs:
                proc.join()

            for k in range(self.n_process):
                i = (self.n_process * g) + k
                f[i, 0] = return_dict1[k]
                f2[i, 0] = return_dict2[k]
                name = "MejorEvo/" + str(self.seed) + ".txt"
                d = open(name, 'a')
                a = np.array([f[i, 0], f2[i, 0], x[i, 0], x[i, 1], x[i, 2], x[i, 3], x[i, 4]])
                np.savetxt(d, a, newline=" ")
                d.write("\n")
                d.close()

            jobs.clear()

        out["F"] = np.column_stack([f, f2])

    def funcion_multiproceso(self, x1, x2, x3, x4, x5, topology, k, return_dict1, return_dict2):
        f1, f2 = ten.clasificacionESN(x1, x2, x3, x4, x5, topology=topology)

        return_dict1[k] = f1
        return_dict2[k] = f2

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict
