import numpy as np


def derivada(net):
    return net * (1 - net)

saida = np.asarray([5, 6, 10, 15, 70, 20])

aplicar_derivada = np.vectorize(derivada)
saida_derivada = aplicar_derivada(saida)
print(saida_derivada)
