import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi, radians, ceil
from copy import copy

X_REFERENCE = 0  # -pi/2
DECAY = 1.0


def show_robot(_pos):
    plt.scatter(_pos[0], _pos[1], color='black')


def rotate_footprint(_pos, _footprint):
    theta = _pos[2] - pi/2
    xy = _pos[0:2]
    rotation = np.asarray([
        [cos(theta), -sin(theta)],
        [sin(theta), cos(theta)]
    ])

    rotated_footprint = copy(_footprint)
    for row in range(0, _footprint.shape[0]):
        rotated_footprint[row, :] = rotation @ rotated_footprint[row, :] + xy
    return rotated_footprint


def show_footprint(_pos, _footprint, _color='green'):
    xy = _pos[0:2]
    rotated_footprint = _footprint  # rotate_footprint(_pos, _footprint)

    arrow = np.asarray([
        rotated_footprint[2, :],
        xy,
        rotated_footprint[3, :]
    ])
    plt.plot(arrow[:, 0], arrow[:, 1], color=_color)

    rotated_footprint = np.append(
        rotated_footprint, rotated_footprint[0, :]).reshape(5, 2)
    plt.plot(rotated_footprint[:, 0], rotated_footprint[:, 1], color=_color)


def show_obstacles(_pos):
    p_size = np.ones_like(_pos[:, 0]) * 250
    plt.scatter(_pos[:, 0], _pos[:, 1], s=p_size, marker='s', color='blue')


def show_path(_path, _robot_footprint):
    plt.plot(_path[:, 0], _path[:, 1], '--', color='black')

    # color = [0.1, 0.3, 1.0]
    # for row in range(0, _path.shape[0]):
    #     color[0] = min(color[0] * 1.6, 1.0)
    #     color[2] = min(color[2] * 0.8, 1.0)
        # if np.linalg.norm(_path[row, :]) > 0:
        #     show_footprint(_path[row, :], _robot_footprint *
        #                    DECAY**(row+1), (color[0], color[1], color[2]))
        # plt.pause(0.3)


def show_all(_robot_pos, _robot_footprint, _obstacles, _path):
    plt.axis('scaled')
    u_lim = 12.5
    l_lim = -2.5
    plt.xlim(l_lim, u_lim)
    plt.ylim(l_lim, u_lim)
    plt.grid(True)

    show_robot(_robot_pos)
    show_footprint(_robot_pos, _robot_footprint)
    show_obstacles(_obstacles)
    show_path(_path, _robot_footprint)
    # plt.show()


def robot_kinematics(_robot, _control, _dt):
    theta = _robot[2]
    B = np.asarray([
        [cos(theta), 0],
        [sin(theta), 0],
        [0, 1]
    ])

    A = np.asarray([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    return A @ _robot + B @ _control * _dt


def simulate_control(_robot, _control, _target, _obstacles, _footprint,
                     _total_time, _dt, _ignore_collision=False):
    path = np.zeros((int(_total_time / _dt), 3))
    target_min_distance = 1e6

    for row in range(0, path.shape[0]):
        _robot = robot_kinematics(_robot, _control, _dt)

        for n in range(0, _footprint.shape[0]):
            point = copy(_robot)
            point[0:2] = _footprint[n, :]
            _footprint[n, :] = robot_kinematics(point, _control, _dt)[0:2]

        if not check_collision(_robot, _footprint, _obstacles) or row < 1 or _ignore_collision:
            path[row, :] = _robot
            target_min_distance = min(target_min_distance,
                                      np.linalg.norm(_target - _control))
        else:
            # print(
            #     f"Collision detected with u = {_control} @ t = {row * _dt} s")
            # return None, -1, -1
            return path[0:row-1, :], target_min_distance, distance_to_obstacles(path[row-1, :], _obstacles)
    return path, target_min_distance, distance_to_obstacles(path[-1, :], _obstacles)


def check_collision(_robot, _footprint, _obstacles):
    # rotate_footprint(_robot, _footprint)
    rotated_footprint = copy(_footprint)
    reference = rotated_footprint[0, :]
    va = rotated_footprint[1, :] - reference
    vb = rotated_footprint[3, :] - reference

    threshold = -0.05

    for row in range(0, _obstacles.shape[0]):
        v_obstacle = _obstacles[row, :] - reference
        inside_a = np.dot(v_obstacle, va) / np.dot(va, va)
        inside_b = np.dot(v_obstacle, vb) / np.dot(vb, vb)

        if threshold <= inside_a <= 1.0 - threshold and threshold <= inside_b <= 1.0 - threshold:
            return True
    return False


def distance_to_obstacles(_pos, _obstacles):
    distance = 1e10
    if _pos.shape[0] > 2:
        xy = _pos[0:2]
    else:
        xy = _pos

    for row in range(0, _obstacles.shape[0]):
        distance = min(distance, np.linalg.norm(_obstacles[row, :] - xy))
    return distance


def find_safe_trajectory(_robot, _linear, _angular, _target, _footprint,
                         _obstacles):
    t_simulation = 10
    t_step = 1

    solution = None
    dist_solution = 1e10
    obst_solution = 0
    control_solution = None

    for v in _linear:
        for w in _angular:
            control = np.asarray([v, w])
            sim_path, target_dist, obst_dist = simulate_control(copy(_robot),
                                                                control,
                                                                _target,
                                                                _obstacles,
                                                                copy(
                                                                    _footprint),
                                                                t_simulation,
                                                                t_step)
            if sim_path is not None and target_dist < dist_solution and obst_dist > 1.9:
                solution = copy(sim_path)
                dist_solution = target_dist
                obst_solution = obst_dist
                control_solution = control
    return solution, dist_solution, obst_solution, control_solution


if __name__ == "__main__":
    # z = [x, y, theta]
    initial_angle = radians(90)
    robot = np.asarray([
        4.64027, -4.82537, 0
    ])
    target_pos = np.asarray([
        0.26, 0.616423
    ])
    # u = [v, w]
    linear_v = list(np.linspace(-0.26, 0.26, 30))
    angular_w = list(np.linspace(-0.2, 0.2, 30))

    robot_footprint = np.asarray([
        [4.93003, -4.62503],
        [4.9305, -5.02502],
        [4.35051, -5.02571],
        [4.35003, -4.62571]
    ])

    obstacles = np.asarray([
        [3.15, -4.25],
        [3.25, -4.25],
        [3.25, -4.15],
        [3.25, -4.05],
        [3.25, -3.95],
        [3.25, -3.85],
        [4.45, -3.65],
        [4.55, -3.65],
        [4.65, -3.65],
        [4.75, -3.65],
        [4.85, -4.65],
        [4.85, -4.55],
        [4.85, -4.45],
        [4.85, -3.65],
        [4.95, -4.95],
        [4.95, -4.85],
        [4.95, -4.75],
        [4.95, -3.55],
        [5.05, -3.55],
        [5.15, -3.55],
        [5.25, -3.55],
        [5.35, -5.15],
        [5.35, -5.05],
        [5.35, -4.85],
        [5.45, -4.85],
        [5.55, -5.35],
        [5.55, -5.25],
        [5.55, -4.85],
        [5.65, -4.95],
        [5.75, -4.95],
        [5.85, -5.65],
        [5.85, -5.55],
        [5.85, -5.45],
        [5.85, -5.35],
        [5.95, -5.35],
        [6.05, -5.35]
    ])


# obstacles[:, 1] = 10 - obstacles[:, 1]
path, target_dist, obst_dist, control = find_safe_trajectory(robot, linear_v,
                                                             angular_w, target_pos,
                                                             robot_footprint,
                                                             obstacles)
plt.title(
    f'[v, $\\omega$] = {control} D = {target_dist:.1f} O = {obst_dist:.1f}')
show_all(robot, robot_footprint, obstacles, path)

d_path, _, _ = simulate_control(robot, target_pos, target_pos, obstacles, robot_footprint, 10, 1, True)
show_path(d_path, robot_footprint)

plt.show()
