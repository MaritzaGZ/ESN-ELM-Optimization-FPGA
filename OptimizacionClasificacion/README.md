# OptimizacionClasificacion

Optimizacion multiobjetivo (NSGA-II) de la ESN de clasificacion definida en
`Original/Hybrid ESN_Classif_MGS.ipynb`. Objetivos: minimizar el numero de
neuronas del reservorio (acotado entre 2 y 50) y maximizar el accuracy
(implementado como minimizar `1 - accuracy`).

Diferencias respecto al notebook original:
- La activacion del reservorio (tanh) se cambio por ReLU, de menor costo
  computacional.
- Se agregaron semillas (`seed_in`, `seed_res`) para poder reproducir la
  evaluacion de un mismo individuo entre corridas.
- No se usa leaking rate (el modelo no tiene integracion "leaky").
- La topologia del reservorio (`full` o `ring`) es seleccionable por linea
  de comandos (ver Uso). En `ring`, cada neurona `i` se conecta solo con la
  `i+1` (ciclo cerrado entre la ultima y la primera); la variable `sparsity`
  no tiene efecto en esta topologia, solo aplica a `full`.

## Datos

`Dataset/Preprocess/` no viene de ninguna fuente externa: se genera con
`preprocess.py` a partir de `Original/Lorenz3D.mat` y `Original/LE_lorenz.mat`
(los mismos archivos que usa el notebook original), aplicando una decimacion
fija y un split train/test fijo, para que todas las evaluaciones de la
busqueda evolutiva usen exactamente los mismos datos.

Las series originales tienen 3000 pasos. Para limitar la red a un maximo de
15-20 entradas se usa `N=150` (3000/150 = 20 features), el extremo superior
del rango 150-200 que cumple esa cota, para retener la mayor informacion
posible.

`Original/Lorenz3D.mat` no esta versionado en git (pesa >100 MB), asi que
debe existir localmente en `../Original/` antes de correr el preprocesamiento.

## Uso

```bash
# 1. Generar el dataset preprocesado (una sola vez)
python3 preprocess.py

# 2. Correr la busqueda evolutiva
python3 NAS-2obj.py <seed> <population> <generations> [topology]

# topology: 'full' (default) o 'ring'

# ejemplo
python3 NAS-2obj.py 1 20 50 ring
```

Mientras corre, se imprime cada individuo evaluado (reservorio, hiperparametros
y accuracy). El resultado queda en `F<seed>.txt` (valores de los objetivos del
frente de Pareto) y `X<seed>.txt` (las variables de diseno correspondientes),
ademas del historial de cada evaluacion en `MejorEvo/<seed>.txt`.

Al terminar:
- Se grafica el frente de Pareto (numero de neuronas vs. accuracy) y se
  guarda en `ParetoFront_<seed>.png`.
- Se elige el mejor resultado (mayor accuracy; en caso de empate, menor
  numero de neuronas), se reentrena y se guarda completo en `MejorModelo/`:
  las matrices `Win.txt`, `bias.txt`, `Wres.txt`, `Wout.txt` y
  `parametros.txt` (reservoir_size, spectral_radius, sparsity, seeds y
  accuracy).
