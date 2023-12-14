import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

from math import pi
import numpy as np

import ssl_simulation_wheel as simulator


def rpm_to_rads(rpm):
    return rpm * 2 * pi / 60


if __name__ == "__main__":
    w_out_required = rpm_to_rads(3000/3)
    max_v = 0
    alpha = 0
    beta = 0

    Z = np.zeros((30, 30))
    X = np.zeros((30, 30))
    Y = np.zeros((30, 30))

    for i, alpha_value in enumerate(range(5, 35, 1)):
        for j, beta_value in enumerate(range(5, 35, 1)):
            # Calcula as componentes para o robô andar em X
            kin, inv_kin = simulator.compute_matrixes(alpha_value, beta_value)
            w_out = simulator.compute_inverse_kinematic(np.asarray([[1], [0.1], [0]]),
                                                        kin)
            # Valor máximo de velocidade angular para o robô andar em X
            max_w_in = w_out_required * w_out / np.linalg.norm(w_out)

            # Velocidade máxima em X com os ângulos alpha e beta
            velocidade = simulator.compute_kinematics(max_w_in, inv_kin)

            Z[i][j] = velocidade[0][0]
            X[i][j] = alpha_value
            Y[i][j] = beta_value
            if velocidade[0][0] > max_v:
                max_v = velocidade[0][0]
                alpha = alpha_value
                beta = beta_value
            # print(alpha_value, beta_value, np.transpose(velocidade))
    # plt.imshow(velocidades)
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    surface = ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, cmap=cm.winter)
    plt.xlabel(r'$\alpha$')
    plt.ylabel(r'$\beta$')
    fig.colorbar(surface, shrink=0.5, aspect=5)

    plt.show()
    print(f'Melhor combinação {max_v}, alpha = {alpha}, beta = {beta}, max = {np.max(Z)}')
