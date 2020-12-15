from math import pi
from Equation import Expression

from sympy import Eq, Function, Poly, Symbol, exp, solve
from sympy.abc import a, b, c, d, x, y, w, z, r, R, A, B
from numpy import float64

if __name__ == "__main__":

    eq1 = (x - a)**2 + (y - b)**2 - r**2
    eq2 = (w - c)**2 + (z - d)**2 - R**2
    eq3 = (x - a)*A + (y - b)*B
    eq4 = (w - c)*A + (z - d)*B
    eq5 = (c - a)**2 + (d - b)**2 - R + r + A**2 + B**2
    eq6 = A - w + x
    eq7 = B - z + y
    ans = solve([eq1, eq2, eq3, eq4, eq5, eq6, eq7], x, y, w, z, A, B, dict=True)
    print(ans)