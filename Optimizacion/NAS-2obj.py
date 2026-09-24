import os
import sys
import time
import numpy
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from evaluate2obj import NetworkOptimization


if __name__ == "__main__":
    seed = int(sys.argv[1])
    population = int(sys.argv[2])
    generations = int(sys.argv[3])

    problem = NetworkOptimization(seed=seed)

    if os.path.isdir('Modelos'):
        print("Borrar")
        os.system('rm -r Modelos')

    os.system('mkdir Modelos')

    if not os.path.isdir('MejorModelo'):
        print("No existe")
        os.system('mkdir MejorModelo')

    if not os.path.isdir('MejorEvo'):
        os.system('mkdir MejorEvo')

    for i in range(population):
        nombre = "mkdir Modelos/Modelo"+str(i)
        os.system(nombre)

    strF = "F"+str(seed)+".txt"
    strX = "X"+str(seed)+".txt"
    algorithm = NSGA2(pop_size=population)
    st = time.time()
    res = minimize(problem,
                   algorithm,
                   ('n_gen',generations),
                   seed=seed,
                   save_history=True,
                   verbose=True)
    et = time.time()
    final=(et-st) / 60
    print("tiempo "+str(final))
    print("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
    numpy.savetxt(strF, res.F )
    numpy.savetxt(strX, res.X )
