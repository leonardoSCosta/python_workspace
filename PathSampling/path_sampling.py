import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi, radians, ceil
from copy import copy

X_REFERENCE = 0  # -pi/2
DECAY = 1.0


def show_robot(_pos):
    plt.scatter(_pos[0], _pos[1], color='black')


def rotate_footprint(_pos, _footprint):
    theta = _pos[2]
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
    rotated_footprint = rotate_footprint(_pos, _footprint)

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


def show_path(_path, _robot_footprint, _control, _robot, _color='black'):
    plt.plot(_path[:, 0], _path[:, 1], '--', color=_color)
    robot = copy(_robot)
    color = [0.1, 0.3, 1.0]
    for row in range(0, _path.shape[0]):
        color[0] = min(color[0] * 1.6, 1.0)
        color[2] = min(color[2] * 0.8, 1.0)
        if np.linalg.norm(_path[row, :]) > 0:
            show_footprint(_path[row, :], _robot_footprint *
                           DECAY**(row+1), (color[0], color[1], color[2]))
    # plt.pause(0.3)


def show_all(_robot_pos, _robot_footprint, _obstacles, _path, _control):
    plt.axis('scaled')
    u_lim = 12.5
    l_lim = -2.5
    plt.xlim(l_lim, u_lim)
    plt.ylim(l_lim, u_lim)
    plt.grid(True)

    show_robot(_robot_pos)
    show_footprint(_robot_pos, _robot_footprint)
    show_obstacles(_obstacles)
    show_path(_path, _robot_footprint, _control, _robot_pos)
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

        if not check_collision(_robot, _footprint, _obstacles) or row < 1 or _ignore_collision:
            path[row, :] = _robot
            target_min_distance = min(target_min_distance,
                                      np.linalg.norm(_target - _control))
        else:
            # print(
            #     f"Collision detected with u = {_control} @ t = {row * _dt} s")
            return None, -1, -1
            # return path[0:row-1, :], target_min_distance, distance_to_obstacles(path[row-1, :], _obstacles)
    print(f"Safe {_control}")
    return path, target_min_distance, distance_to_obstacles(path[-1, :], _obstacles)


def check_collision(_robot, _footprint, _obstacles):
    rotated_footprint = rotate_footprint(_robot, copy(_footprint))
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
            if sim_path is not None and target_dist < dist_solution and obst_dist > 0.0:
                solution = copy(sim_path)
                dist_solution = target_dist
                obst_solution = obst_dist
                control_solution = control
    return solution, dist_solution, obst_solution, control_solution


if __name__ == "__main__":
    # z = [x, y, theta]
    robot = np.asarray([
        6.94887, 0.805362, 0.111927
    ])
    target_pos = np.asarray([
        0.00, -0.572404
    ])
    # u = [v, w]
    linear_v = list(np.linspace(-0.26, 0.26, 30))
    angular_w = list(np.linspace(-1.0, 1.0, 30))

    robot_footprint = np.asarray([
        [0.38, 0.29],
        [0.38, -0.29],
        [-0.38, -0.29],
        [-0.38, 0.29]
    ])

    obstacles = np.asarray([
        [5.825, -0.125],
        [5.875, -0.025],
        [5.925, 0.025],
        [5.925, 0.275],
        [5.925, 0.425],
        [5.975, 0.525],
        [6.025, 0.575],
        [6.075, 0.575],
        [6.125, -0.375],
        [6.125, 0.625],
        [6.175, 0.625],
        [6.225, -0.325],
        [6.225, 0.575],
        [6.275, -0.275],
        [6.275, -0.225],
        [6.275, 0.675],
        [6.325, -0.175],
        [6.325, -0.125],
        [6.325, 0.725],
        [6.375, -0.075],
        [6.375, 0.775],
        [6.425, 0.775],
        [6.475, -0.025],
        [6.475, 0.825],
        [6.525, 0.825],
        [6.575, 0.825],
        [6.625, 0.775],
        [6.675, 0.775],
        [6.725, 0.725],
        [6.725, 1.425],
        [6.775, 0.625],
        [6.775, 0.675],
        [6.775, 1.325],
        [6.825, -0.175],
        [6.825, 0.425],
        [6.825, 0.475],
        [6.825, 0.525],
        [6.825, 0.575],
        [6.825, 1.275],
        [6.875, 0.375],
        [6.875, 1.225],
        [6.925, -0.125],
        [6.925, 0.375],
        [6.925, 1.175],
        [6.975, -0.075],
        [6.975, 0.325],
        [6.975, 1.025],
        [7.025, 0.325],
        [7.025, 0.925],
        [7.025, 0.975],
        [7.025, 1.875],
        [7.075, 0.275],
        [7.075, 0.875],
        [7.075, 1.925],
        [7.125, 0.175],
        [7.125, 0.825],
        [7.125, 1.975],
        [7.175, 0.775],
        [7.225, 2.025],
        [7.275, 0.725],
        [7.375, 0.675],
        [7.375, 0.725],
        [7.425, 0.625],
        [7.425, 1.975],
        [7.475, 0.575],
        [7.475, 1.325],
        [7.525, 0.525],
        [7.525, 1.325],
        [7.575, 0.475],
        [7.575, 1.325],
        [7.625, 0.425],
        [7.625, 1.325],
        [7.675, 0.375],
        [7.675, 1.325],
        [7.675, 1.875],
        [7.725, 0.325],
        [7.725, 1.325],
        [7.775, 0.275],
        [7.775, 1.325],
        [7.825, 0.225],
        [7.825, 1.375],
        [7.875, 0.175],
        [7.875, 1.375],
        [7.875, 1.875],
        [7.925, 0.125],
        [7.925, 1.375],
        [7.975, 0.075],
        [7.975, 0.825],
        [7.975, 1.425],
        [7.975, 1.925],
        [8.025, 0.025],
        [8.025, 0.825],
        [8.025, 1.425],
        [8.075, -0.025],
        [8.075, 0.775],
        [8.075, 1.475],
        [8.125, -0.075],
        [8.125, 0.725],
        [8.125, 1.525],
        [8.175, -0.125],
        [8.175, 0.675],
        [8.175, 1.625],
        [8.225, 0.675],
        [8.275, -0.175],
        [8.275, 0.625],
        [8.325, 0.225],
        [8.325, 0.525],
        [8.375, 0.175],
        [8.375, 0.425],
        [8.425, 0.125]
    ])

# obstacles[:, 1] = 10 - obstacles[:, 1]
path, target_dist, obst_dist, control = find_safe_trajectory(robot, linear_v,
                                                             angular_w, target_pos,
                                                             robot_footprint,
                                                             obstacles)
plt.title(
    f'[v, $\\omega$] = {control} D = {target_dist:.1f} O = {obst_dist:.1f}')
# show_all(robot, robot_footprint, obstacles, path, control)

d_path, _, _ = simulate_control(
    robot, target_pos, target_pos, obstacles, robot_footprint, 10, 1, True)
# show_path(d_path, robot_footprint, target_pos, robot, 'blue')
show_all(robot, robot_footprint, obstacles, d_path, target_pos)

plt.show()
