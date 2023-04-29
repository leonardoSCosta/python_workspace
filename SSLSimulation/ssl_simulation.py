import numpy as np
from numpy.linalg import inv
from math import sin, cos, pi

import matplotlib.pyplot as plt

WHEEL_R = 54e-3/2
THETA = 1 * pi / 180
T_STEP = 0.01

KINEMATICS = np.asarray([[-sin(THETA), -cos(THETA), 1],
                         [sin(THETA), - cos(THETA), 1],
                         [sin(THETA), cos(THETA), 1],
                         [-sin(THETA), cos(THETA), 1]])

INV_KINEMATICS = np.transpose(np.asarray([[1/-sin(THETA), 1/sin(THETA), 1/sin(THETA), 1/-sin(THETA)],
                                          [1/-cos(THETA), 1/-cos(THETA), 1/cos(THETA), 1/cos(THETA)],
                                          [0, 0, 0, 0]]))


def compute_inverse_kinematic(v_in: np.array):
    w_out = np.zeros((4, 1))

    w_out = 1 / WHEEL_R * KINEMATICS @ v_in  # 5170 rpm 5170*2*3.1415/60 = 181 rad/s
#      print(w_out)
    return w_out


def compute_kinematics(w_in: np.array):
    w_out = np.zeros((3, 1))
    w_out = WHEEL_R * w_in @ INV_KINEMATICS
    return w_out / 4


if __name__ == "__main__":
    v_in = np.transpose(np.asarray([[2, 0, 0]], dtype='f4'))
    time = np.asarray([t/100 for t in range(0, 1000, 1)])

    setpoint = np.asarray([[0], [0], [0]], dtype='f4')
    pos = np.asarray([[0], [0], [pi/2]], dtype='f4')
    real_trajectory = np.zeros((1000, 2))
    trajectory = np.zeros((1000, 2))

    for n, t in enumerate(time):
        w_out = compute_inverse_kinematic(v_in)
#          w_out += np.transpose(np.asarray([[5, 20, 5, 20]]))
        w_out = np.ones_like(w_out) * pi * 100

        v_out = compute_kinematics(np.transpose(w_out))

        setpoint = setpoint + v_in * T_STEP
        trajectory[n, :] = np.transpose(setpoint[0:2])

        pos = pos + np.transpose(v_out) * T_STEP
        real_trajectory[n, :] = np.transpose(pos[0:2])

#          if abs(pos[0][0] - 3) < 0.01:
#              print(pos, t)
#              break

    print(v_out, w_out)
#      plt.plot(trajectory[:, 0], trajectory[:, 1])
    plt.plot(real_trajectory[:, 0], real_trajectory[:, 1])
    plt.legend(["Trajetória", "Trajetória Real"])
    plt.show()
