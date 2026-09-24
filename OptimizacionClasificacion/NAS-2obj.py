import os
import sys
import time
import numpy
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from evaluate2obj import NetworkOptimization
import TrainEvalNet as ten


if __name__ == "__main__":
    seed = int(sys.argv[1])
    population = int(sys.argv[2])
    generations = int(sys.argv[3])
    topology = sys.argv[4] if len(sys.argv) > 4 else 'full'

    problem = NetworkOptimization(seed=seed, topology=topology)

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

    # ------------------------------------------------------------------
    # Frente de Pareto: reservoir_size vs accuracy
    # ------------------------------------------------------------------
    F = numpy.atleast_2d(res.F)
    X = numpy.atleast_2d(res.X)
    accuracy = 1.0 - F[:, 0]
    reservorios = F[:, 1]

    print("\nFrente de Pareto (reservorio, accuracy):")
    orden = numpy.argsort(reservorios)
    for idx in orden:
        print(f"  reservorio={int(reservorios[idx]):4d}  accuracy={accuracy[idx]:.4f}")

    plt.figure(figsize=(8, 5))
    plt.scatter(reservorios, accuracy, c='tab:blue')
    plt.xlabel('Numero de neuronas del reservorio')
    plt.ylabel('Accuracy')
    plt.title(f'Frente de Pareto (seed={seed}, topologia={topology})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"ParetoFront_{seed}.png")
    plt.show()

    # ------------------------------------------------------------------
    # Mejor resultado: mayor accuracy, desempate por menor reservorio.
    # Se reentrena para poder guardar las matrices generadas.
    # ------------------------------------------------------------------
    mejor_idx = numpy.lexsort((reservorios, -accuracy))[0]
    x1, x2, x3, x4, x5 = X[mejor_idx]
    x3, x4, x5 = int(x3), int(x4), int(x5)

    print(f"\nMejor resultado: reservorio={x3}  accuracy={accuracy[mejor_idx]:.4f}")
    print("Reentrenando para guardar el modelo completo en MejorModelo/ ...")

    esn, acc_final = ten.entrenar_modelo(x1, x2, x3, x4, x5, topology=topology)
    esn.guardar_modelo("MejorModelo", accuracy=acc_final)
    print("Modelo guardado en MejorModelo/ (matrices Win/bias/Wres/Wout + parametros.txt)")
