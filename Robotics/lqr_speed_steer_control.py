#!/usr/bin/python3.8
"""

Path tracking simulation with LQR speed and steering control

author Atsushi Sakai (@Atsushi_twi)

"""
import cubic_spline_planner

import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

N_STATES = 4
N_INPUTS = 4

# === Parameters =====

# LQR parameter
#  lqr_Q = np.zeros((4, 4))
#  lqr_Q[0, 0] = 1000  # e
#  lqr_Q[1, 1] = 100  # theta
#  lqr_Q[2, 2] = 100  # deltaV
#  lqr_Q[3, 3] = 1  # f

lqr_Q = np.eye(N_STATES)
#  lqr_Q[0, 0] = 1  # e
#  lqr_Q[1, 1] = 1  # theta
#  lqr_Q[2, 2] = 1  # deltaV
#  lqr_Q[3, 3] = 0.01  # f

#  lqr_Q = np.asarray(
#  [[3.52279292,-8.8131733,-16.50164272,8.73445083],
#  [13.94328912,-28.03779627,3.08815918,-2.10678931],
#  [-10.58290561,-28.22386331,-6.43919178,-4.27962428],
#  [-18.40922145,11.21123129,-4.76972877,29.89162698]],
#  )

#  lqr_R = np.zeros((3, 3))
#  lqr_R[0, 0] = 0.01  # vx
#  lqr_R[1, 1] = 1  # vw
#  lqr_R[2, 2] = 0.0001  # ay

lqr_R = np.eye(N_INPUTS) / 100

#  lqr_R = np.asarray(
#  [[-16.52277903,-8.73471333,0.07317548],
#  [2.42333861,18.64435174,-16.4625695],
#  [23.19449441,0.35538129,16.49761124]]
#  )

dt = 0.015  # time tick[s]
close = False
show_animation = False

USE_DERIVATIVES = False


class State:
    def __init__(self, x=0.0, y=0.0, w=0.0, v=0.0):
        self.x = x
        self.y = y
        self.theta = 0.0

        self.vx = 0
        self.vy = 0
        self.v = v
        self.w = w


def update(state, vx, vy, vw, a):
    # Velocidades locais
    state.vy = max(min(2, vy), -2)
    state.vx = max(min(2, vx), -2)
    state.w = max(min(6.5, vw), -6.5)

#      state.vy = state.vy - vy * dt
#      state.vx = vx
#      state.w = vw

    # Velocidade global
    state.v = state.v + a * dt

    dir = np.asarray([state.vx, state.vy])
    dir = dir / np.linalg.norm(dir) * state.v
    state.vx = dir[0]
    state.vy = dir[1]

    Vx = state.vx # state.vx * math.sin(state.theta) + state.vy * math.cos(state.theta)
    Vy = state.vy # -state.vx * math.cos(state.theta) + state.vy * math.sin(state.theta)

    state.x = state.x + Vx * dt
    state.y = state.y + Vy * dt
    state.theta = state.theta + state.w * dt
    while state.theta > math.pi * 2:
        state.theta = state.theta - math.pi * 2

#      state.v = math.sqrt(Vx**2 + Vy**2) * Vy / abs(Vy+1e-10)
#      state.v = math.sqrt(state.vx**2 + state.vy**2)

    return state


def pi_2_pi(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def solve_dare(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    x = Q
    x_next = Q
    max_iter = 10
    eps = 0.01

    i = 0
    for i in range(max_iter):
        # @ = Matrix multiplication
        # * = Element-wise multiplication
        x_next = A.T @ x @ A - A.T @ x @ B @ \
                 la.inv(R + B.T @ x @ B) @ B.T @ x @ A + Q
        if (abs(x_next - x)).max() < eps:
            break
        x = x_next

    return x_next


def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    x[k+1] = A x[k] + B u[k]
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    # ref Bertsekas, p.151
    """

    # first, try to solve the ricatti equation
    X = solve_dare(A, B, Q, R)

    # compute the LQR gain
    K = la.inv(B.T @ X @ B + R) @ (B.T @ X @ A)

    eig_result = la.eig(A - B @ K)

    return K, X, eig_result[0]


def lqr_speed_steering_control(state, cx, cy, cw, ck, pe, pth_e, sp, Q, R):
    ind, e = calc_nearest_index(state, cx, cy, cw)

    ind = ind + 1
    ind = min(ind, len(cx)-1)

    f_ind = ind + 5
    f_ind = min(f_ind, len(cx)-1)

    f = math.sqrt((state.x - cx[f_ind])**2 + (state.y - cy[f_ind])**2)

    v = state.v
    tv = sp[ind]
    th_e = pi_2_pi(cw[ind] - state.theta)

    A = np.eye(N_STATES)

    B = np.zeros((N_STATES, N_INPUTS))
    B[0, 0] = dt
    B[1, 1] = dt
    B[2, 2] = dt
    B[3, 3] = dt
#      B = np.zeros((4, 3))
#      B[0, 0] = -math.cos(th_e) * dt
#      B[0, 2] = -math.sin(th_e) * dt * dt
#      B[1, 1] = -dt
#      B[2, 2] = dt
#      B[3, 2] = -dt * dt

    K, _, _ = dlqr(A, B, Q, R)
    # state vector
    # x = [e, th_e, delta_v]
    # e: lateral distance to the path
    # th_e: angle difference to the path
    # delta_v: difference between current speed and target speed
#      x = np.zeros((4, 1))
#      x[0, 0] = e
#      x[1, 0] = th_e
#      x[2, 0] = v - tv
#      x[3, 0] = f
    x = np.zeros((N_STATES, 1))
    x[0, 0] = state.x - cx[ind]
    x[1, 0] = state.y - cy[ind]
    x[2, 0] = state.theta - cw[ind]
    x[3, 0] = state.v - sp[ind]
    print(K)
#      x = np.zeros((4, 1))
#      x[0, 0] = e
#      x[1, 0] = th_e
#      x[2, 0] = v - tv
#      x[3, 0] = f

    # input vector
    # u = [vx, omega, ay]
    # vx: lateral velocity
    # omega: angular velocity
    # ay: longitudinal acceleration
    ustar = -K @ x


    # calc input
#      vx = ustar[0, 0]
#      vw = ustar[1, 0]
#      vy = ustar[2, 0]
    vx = ustar[0, 0]
    vy = ustar[1, 0]
    vw = ustar[2, 0]
    ac = ustar[3, 0]

    cost = np.transpose(x) @ lqr_Q @ x + np.transpose(ustar) @ lqr_R @ ustar

    return ind, e, th_e, f, vx, vy, vw, ac, cost


def calc_nearest_index(state, cx, cy, cw):
    dx = [state.x - icx for icx in cx]
    dy = [state.y - icy for icy in cy]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind)

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def close_program(_close: bool):
    global close
    close = _close


def do_simulation(cx, cy, cw, ck, speed_profile, goal):
    T = 42.0  # max simulation time
    goal_dis = 0.1
    stop_speed = 1.0

    state = State(x=-1.5, y=1.12, w=0.0, v=0.0)

    time = 0.0
    x = [state.x]
    y = [state.y]
    w = [state.w]
    v = [state.v]
    t = [0.0]
    vax = [state.vx]
    vay = [state.vy]
    vcost = [0]
    vtheta = [0]
    ve = [0]
    vf = [0]

    e, e_th = 0.0, 0.0

    while T >= time:
        global close
        if close:
            break

        target_ind, e, e_th, f, vx, vy, vw, vt, cost = lqr_speed_steering_control(
            state, cx, cy, cw, ck, e, e_th, speed_profile, lqr_Q, lqr_R)

        state = update(state, vx, vy, vw, vt)

        # Disturbs
#          if time > 2 and time < 2.2:
#              state.x += -1.4
#              state.y -= 0.5
#          if time > 10 and time < 10.2:
#              state.x += -3.4
#              state.y -= 2.5

#          if abs(state.v) <= stop_speed:
#              target_ind += 1

        time = time + dt

        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        if math.hypot(dx, dy) <= goal_dis:
            print("Goal")
            break

        x.append(state.x)
        y.append(state.y)
        w.append(state.w)
        v.append(state.v)
        vax.append(state.vx)
        vay.append(state.vy)
        vcost.append(cost[0][0])
        vtheta.append(e_th)
        ve.append(e)
        vf.append(f)
        t.append(time)

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [close_program(True) if event.key == 'escape' else None])
            plt.plot(cx, cy, "-r", label="course")
            plt.plot(x, y, "ob", label="trajectory")
            plt.plot(cx[target_ind], cy[target_ind], "xg", label="target")
            plt.plot([state.x, state.x + math.cos(state.theta) * 0.5],
                     [state.y, state.y + math.sin(state.theta) * 0.5], "-b")
            plt.axis("equal")
            plt.grid(True)
            plt.title("speed[m/s]:" + str(round(math.sqrt(state.vx**2 + state.vy**2), 2))
                      + ",target index:" + str(target_ind))
            plt.pause(0.000001)

    return t, x, y, w, v, vax, vay, vcost, vtheta, ve, vf


def calc_speed_profile(cw, target_speed):
    speed_profile = [target_speed] * len(cw)

    direction = 1.0

    # Set stop point
    for i in range(len(cw) - 1):
        dw = abs(cw[i + 1] - cw[i])
        switch = math.pi / 4.0 <= dw < math.pi / 2.0

        if switch:
            direction *= -1

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

        if switch:
            speed_profile[i] = 0.0

    # speed down
    for i in range(min(40, len(speed_profile))):
        speed_profile[-i] = target_speed / (50.0 - i)
        if speed_profile[-i] <= 1.0:
            speed_profile[-i] = 1.0

    return speed_profile


def main():
    print("LQR steering control tracking start!!")
#      ax = [0.0, 1.0]
#      ay = [5.0, 8.0]
    ax = [1.00, 1.40, 1.60, 2.25, 2.00, 2.75, 3.00, 3.50, 4.00]
    ay = [-0.50, .00, -0.30, -0.50, 0.65, 0.30, 0.00, 0.00, 1.00]
#      ax = [0.0, 5.0, 5.0, 0.0, 2.0]
#      ay = [0.0, 0.0, 5.0, 5.0, 0.0]
    goal = [ax[-1], ay[-1]]

    cx, cy, cw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)
    target_speed = 2.0  # simulation parameter km/h -> m/s

    sp = calc_speed_profile(cw, target_speed)

#      cx = (np.asarray([-1499.09, -1339.53, -1196.5, -1069.99, -960.013, -866.556, -789.623,
#            -729.215, -685.331, -657.971, -647.136, -629.484, -611.832, -594.18,
#            -576.527, -558.875, -541.223, -523.571, -505.918, -488.266, -470.614,
#            -452.962, -435.309, -417.657, -400.005, -382.353, -364.7, -347.048,
#            -329.396, -311.744, -294.091, -276.439, -258.787, -241.135, -223.482,
#            -205.83, -188.178, -170.526, -152.873, -135.221, -117.569, -99.9166,
#            -82.2643, -64.6121, -46.9598, -29.3076, -11.6553, 5.99692, 23.6492,
#            41.3014, 58.9537, 76.6059, 94.2582, 111.91, 129.563, 147.215,
#            164.867, 182.519, 200.172, 217.824, 235.476, 253.128, 270.781,
#            288.433, 306.085, 323.737, 341.39, 359.042, 376.694, 394.346,
#            411.999, 429.651, 447.303, 464.955, 482.608, 500.26, 517.912,
#            535.564, 553.217, 570.869, 588.521, 606.173, 623.826, 641.478,
#            659.13, 676.782, 694.434, 712.087, 729.739, 747.391, 765.043,
#            782.696, 800.348, 803.769, 806.97, 809.953, 812.717, 815.261,
#            817.587, 819.693, 821.581, 823.249, 824.698, 854.639, 884.58,
#            914.521, 944.462, 974.402, 1004.34, 1034.28, 1064.22, 1094.17,
#            1124.11, 1154.05, 1183.99, 1213.93, 1243.87, 1273.81, 1303.75,
#            1333.69, 1363.63, 1393.57, 1423.51, 1453.46, 1483.4, 1513.34,
#            1543.28, 1573.22, 1603.16, 1633.1, 1663.04, 1692.98, 1722.92,
#            1752.86, 1782.8, 1812.74, 1842.69, 1872.63, 1902.57, 1932.51])/1000).tolist()
#      cy = (np.asarray([1120, 1020.26, 930.653, 851.169, 781.812, 722.581, 673.477, 634.499,
#            605.648, 586.923, 578.324, 554.068, 529.811, 505.554, 481.297,
#            457.04, 432.783, 408.526, 384.269, 360.012, 335.755, 311.498,
#            287.241, 262.984, 238.728, 214.471, 190.214, 165.957, 141.7, 117.443,
#            93.1861, 68.9291, 44.6722, 20.4153, -3.84162, -28.0985, -52.3554,
#            -76.6124, -100.869, -125.126, -149.383, -173.64, -197.897, -222.154,
#            -246.411, -270.668, -294.925, -319.182, -343.438, -367.695, -391.952,
#            -416.209, -440.466, -464.723, -488.98, -513.237, -537.494, -561.751,
#            -586.008, -610.265, -634.521, -658.778, -683.035, -707.292, -731.549,
#            -755.806, -780.063, -804.32, -828.577, -852.833, -877.09, -901.347,
#            -925.604, -949.861, -974.118, -998.375, -1022.63, -1046.89, -1071.15,
#            -1095.4, -1119.66, -1143.92, -1168.17, -1192.43, -1216.69, -1240.94,
#            -1265.2, -1289.46, -1313.72, -1337.97, -1362.23, -1386.49, -1410.74,
#            -1415.35, -1419.46, -1423.08, -1426.2, -1428.83, -1430.97, -1432.61,
#            -1433.76, -1434.42, -1434.58, -1432.7, -1430.81, -1428.93, -1427.05,
#            -1425.16, -1423.28, -1421.39, -1419.51, -1417.63, -1415.74, -1413.86,
#            -1411.98, -1410.09, -1408.21, -1406.33, -1404.44, -1402.56, -1400.68,
#            -1398.79, -1396.91, -1395.03, -1393.14, -1391.26, -1389.38, -1387.49,
#            -1385.61, -1383.73, -1381.84, -1379.96, -1378.08, -1376.19, -1374.31,
#            -1372.43, -1370.54, -1368.66, -1366.78, -1364.89])/1000).tolist()
#      cw = [-0.558666, -0.559685, -0.560969, -0.562634, -0.564884, -0.568089,
#            -0.573021, -0.581593, -0.600179, -0.670813, -0.941708, -0.941708,
#            -0.941708, -0.941708, -0.941708, -0.941708, -0.941708, -0.941707,
#            -0.941708, -0.941708, -0.941708, -0.941708, -0.941708, -0.941707,
#            -0.941707, -0.941707, -0.941707, -0.941707, -0.941707, -0.941707,
#            -0.941707, -0.941707, -0.941707, -0.941707, -0.941707, -0.941707,
#            -0.941707, -0.941707, -0.941707, -0.941707, -0.941707, -0.941707,
#            -0.941707, -0.941707, -0.941707, -0.941708, -0.941708, -0.941708,
#            -0.941708, -0.941708, -0.941708, -0.941708, -0.941708, -0.941708,
#            -0.941708, -0.941707, -0.941707, -0.941707, -0.941707, -0.941707,
#            -0.941707, -0.941707, -0.941707, -0.941707, -0.941707, -0.941707,
#            -0.941707, -0.941707, -0.941707, -0.941707, -0.941707, -0.941707,
#            -0.941707, -0.941707, -0.941707, -0.941707, -0.941708, -0.941709,
#            -0.941709, -0.941709, -0.941709, -0.941709, -0.941709, -0.941709,
#            -0.941709, -0.941709, -0.941709, -0.941709, -0.941709, -0.941709,
#            -0.941709, -0.941709, -0.931849, -0.909092, -0.881258, -0.846507,
#            -0.80198, -0.743132, -0.662438, -0.547117, -0.374766, -0.111646,
#            0.0628221, 0.0628221, 0.0628221, 0.0628221, 0.0628221, 0.0628221,
#            0.0628221, 0.0628221, 0.0628221, 0.0628221, 0.0628221, 0.0628221,
#            0.0628221, 0.0628221, 0.0628221, 0.0628221, 0.0628221, 0.0628221,
#            0.0628221, 0.0628221, 0.0628221, 0.0628221, 0.0628221, 0.0628221,
#            0.0628221, 0.0628221, 0.0628221, 0.0628221, 0.0628221, 0.0628221,
#            0.0628221, 0.0628221, 0.0628221, 0.0628221, 0.0628221, 0.0628221,
#            0.0628221, 0.0628221]
#      sp = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
#            1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
#            1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
#            1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
#            1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
#            1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
#            1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
#            1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
#            1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5,
#            1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
#      goal = [cx[-1], cy[-1]]

    t, x, y, w, v, vax, vay, cost, vtheta, ve, vf = do_simulation(cx, cy, cw, ck, sp, goal)

    if True:  # pragma: no cover
        plt.close()
        plt.figure()
        plt.plot(ax, ay, "xb", label="waypoints")
        plt.plot(cx, cy, "-r", label="target course")
        plt.plot(x, y, "-g", label="tracking")
        plt.grid(True)
        plt.axis("equal")
        plt.xlabel("x[m]")
        plt.ylabel("y[m]")
        plt.legend()

#          plt.figure()
#          plt.plot(cx, "xb", label="X Path")
#          plt.plot(x, "xr", label="X True")
#          plt.legend()
#          plt.figure()
#          plt.plot(cy, "xb", label="Y Path")
#          plt.plot(y, "xr", label="Y True")
#          plt.legend()

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(vax, "-r", label="Velocity x")
        plt.plot(vay, "-b", label="Velocity y")
        plt.grid(True)
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("Velocity [m/s]")

        plt.subplot(2, 1, 2)
        plt.plot(cost, "-r", label="cost")
        plt.grid(True)
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("J")

        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(vtheta, "-r", label="$\\theta$")
        plt.grid(True)
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("$\\theta$ [rad]")

        plt.subplot(3, 1, 2)
        plt.plot(ve, "-r", label="e")
        plt.grid(True)
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("e [m]")

        plt.subplot(3, 1, 3)
        plt.plot(vf, "-b", label="f")
        plt.grid(True)
        plt.legend()
        plt.xlabel("k")
        plt.ylabel("f [m]")

        plt.figure()
        plt.plot(sp)

        plt.show()


if __name__ == '__main__':
    show_animation = True
    main()
