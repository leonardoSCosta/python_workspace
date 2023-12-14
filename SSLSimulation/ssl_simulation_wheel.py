#!/usr/bin/python3.8
import numpy as np
from math import sin, cos, pi, hypot, radians
import matplotlib.pyplot as plt

WHEEL_R = 54e-3/2
WHEEL_D = 72e-3
# THETA = 33 * pi / 180
T_STEP = 0.01


def compute_matrixes(alpha, beta):
    # print(alpha)
    # print(beta)
    # alpha = alpha + 1e-10
    # beta = beta + 1e-10
    #      INV_KINEMATICS = np.transpose(
    #          np.asarray([[1/-sin(alpha), 1/sin(beta), 1/sin(beta), 1/-sin(alpha)],
    #                      [1/-cos(alpha), 1/-cos(beta), 1/cos(beta), 1/cos(alpha)],
    #                      [0, 0, 0, 0]]))
    # alpha - frente - phi
    # beta - tras - theta

#      INV_KINEMATICS = np.asarray(
#                      [[-1/(sin(beta) + sin(alpha)), -1/(sin(beta) + sin(alpha)),
#                        1/(sin(beta) + sin(alpha)), 1/(sin(beta) + sin(alpha))],
#                       [cos(alpha)/(cos(beta)**2 + cos(alpha)**2),
#                        -cos(alpha)/(cos(beta)**2 + cos(alpha)**2),
#                        -cos(beta)/(cos(beta)**2 + cos(alpha)**2),
#                        cos(beta)/(cos(beta)**2 + cos(alpha)**2)],
#                       [sin(beta)/(sin(beta) + sin(alpha)),
#                        sin(beta)/(sin(beta) + sin(alpha)),
#                        sin(alpha)/(sin(beta) + sin(alpha)),
#                        sin(alpha)/(sin(beta) + sin(alpha))]])

#      KINEMATICS = np.asarray([[-sin(alpha), cos(alpha), 1],
#                              [-sin(pi - alpha), cos(pi - alpha), 1],
#                              [-sin(pi + beta), cos(pi + beta), 1],
#                              [-sin(2*pi - beta), cos(2*pi - beta), 1]])
    KINEMATICS = np.asarray([[-sin(alpha), cos(alpha), WHEEL_D],
                            [-sin(beta), -cos(beta), WHEEL_D],
                            [sin(beta), -cos(beta), WHEEL_D],
                            [sin(alpha), cos(alpha), WHEEL_D]])
    INV_KINEMATICS = np.linalg.pinv(KINEMATICS)
    return KINEMATICS, INV_KINEMATICS


def compute_inverse_kinematic(v_in: np.array, kinematics_matrix):
    w_out = np.zeros((4, 1))
    w_out = 1 / WHEEL_R * kinematics_matrix @ v_in
    return w_out


def compute_kinematics(w_in: np.array, inverse_kinematics_matrix):
    v_out = np.zeros((3, 1))
    v_out = WHEEL_R * inverse_kinematics_matrix @ w_in
    return v_out


def do_simulation(vel_x, alpha, beta, goal_position):
    w_1 = []
    w_2 = []
    w_3 = []
    w_4 = []
    dt = []
    goal_point_x = goal_position[0]
    goal_point_y = goal_position[1]
    kinematics, inverse_kinematics = compute_matrixes(alpha, beta)
    v_in = np.transpose(np.asarray([[vel_x, 0, 0]], dtype='f4'))
    time = np.asarray([t/100 for t in range(0, 1000, 1)])

    # setpoint = np.asarray([[0], [0], [0]], dtype='f4')
    pos = np.asarray([[0], [0], [0]], dtype='f4')
    real_trajectory = np.zeros((1000, 2))
    # trajectory = np.zeros((1000, 2))

    for n, t in enumerate(time):
        w_out = compute_inverse_kinematic(v_in, kinematics)
        # print(w_out)
        w_1.append(w_out[0, 0])
        w_2.append(w_out[1, 0])
        w_3.append(w_out[2, 0])
        w_4.append(w_out[3, 0])
        dt.append(t)

        v_out = compute_kinematics(w_out, inverse_kinematics)

        pos = pos + v_out * T_STEP
        real_trajectory[n, :] = np.transpose(pos[0:2])
        if hypot(pos[0] - goal_point_x, pos[1] - goal_point_y) <= 0.1:
            # print("Goal")
            # print(t)
            break

    # plt.plot(trajectory[:, 0], trajectory[:, 1])
#      plt.plot(real_trajectory[:, 0], real_trajectory[:, 1])
#      plt.show()

    return w_1[0], w_2[0], w_3[0], w_4[0]


if __name__ == "__main__":
    v_in = np.transpose(np.asarray([[1, 0.5, 0]], dtype='f4'))
    time = np.asarray([t/100 for t in range(0, 100, 1)])

    setpoint = np.asarray([[0], [0], [0]], dtype='f4')
    pos = np.asarray([[0], [0], [0]], dtype='f4')

    real_trajectory = np.zeros((1000, 2))
    trajectory = np.zeros((1000, 2))
#      alpha = 0.06981317007977318
#      beta = 0.017453292519943295
    alpha = radians(5)
    beta = radians(1)

    kin, inv_kin = compute_matrixes(alpha, beta)
    w_out_required = 1/3 * 3000 * 2 * pi / 60
    for n, t in enumerate(time):
        w_out = compute_inverse_kinematic(v_in, kin)

        max_w_in = w_out_required * w_out / np.linalg.norm(w_out)
        v_out = compute_kinematics(max_w_in, inv_kin)

        pos = pos + v_out * T_STEP
        real_trajectory[n, :] = np.transpose(pos[0:2])

    print(np.transpose(w_out), np.sum(abs(w_out)), v_out)

    # plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.plot(real_trajectory[:, 0], real_trajectory[:, 1])
    plt.legend(["Trajetória", "Trajetória Real"])
    plt.show()
