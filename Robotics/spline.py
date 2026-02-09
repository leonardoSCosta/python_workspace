import numpy as np
import matplotlib.pyplot as plt
from math import atan2

from cubic_spline_planner import CubicSpline1D
from cubic_spline_planner import CubicSpline2D

x = [-2.5, 0.0, 2.5, 5.0, 7.5, 6.0, 5.0]
y = [0.7, -2, 5, 6.5, 0.0, -2.0, 2.0]
ds = 0.1  # [m] distance of each interpolated points
sp = CubicSpline2D(x, y)
s = np.arange(0, sp.s[-1], ds)
rx, ry, ryaw, rk = [], [], [], []

for i_s in s:
    ix, iy = sp.calc_position(i_s)
    rx.append(ix)
    ry.append(iy)
    ryaw.append(sp.calc_yaw(i_s))
    rk.append(sp.calc_curvature(i_s))

myyaw = []
for i in range(len(rx)-1):
    myyaw.append(atan2(ry[i+1] - ry[i], rx[i+1] - rx[i]))
myyaw.append(myyaw[-1])

plt.subplots(1)
plt.plot(x, y, "xb", label="Data points")
plt.plot(rx, ry, "-r", label="Cubic spline path")
plt.grid(True)
plt.axis("equal")
plt.xlabel("x[m]")
plt.ylabel("y[m]")
plt.legend()

plt.subplots(1)
plt.plot(s, [np.rad2deg(iyaw) for iyaw in ryaw], "-r", label="yaw")
# plt.plot(s, [np.rad2deg(iyaw) for iyaw in myyaw], "-b", label="my yaw")
plt.grid(True)
plt.legend()
plt.xlabel("line length[m]")
plt.ylabel("yaw angle[deg]")

plt.subplots(1)
plt.plot(s, rk, "-r", label="curvature")
plt.grid(True)
plt.legend()
plt.xlabel("line length[m]")
plt.ylabel("curvature")

plt.show()
