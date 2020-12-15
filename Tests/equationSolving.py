from math import pi
from Equation import Expression

from sympy import Eq, Function, Poly, Symbol, exp, solve
from sympy.abc import a, b, c, d, x, y, w, z, r, R
from numpy import float64

# def FA(F=0,A=0):
#     if F == 0:
#         F = Symbol('F')
#     if A == 0:
#         A = Symbol('A')

#     return F/A


if __name__ == "__main__":
    # wtt= Function('Wtt')
    # wtc= Function('Wtc')
    # wtr= Function('Wtr')

    # b = Symbol('b')
    # d = Symbol('d')
    # h = Symbol('h')

    # wt = Symbol('Wt')
    
    eq1 = (x - a)**2 + (y - b)**2 - r**2
    eq2 = (w - c)**2 + (z - d)**2 - R**2
    eq3 = (x - a)*(w-x) + (y - b)*(z-y)
    eq4 = (w - c)*(w-x) + (z - d)*(z-y)
    eq5 = (c - a)**2 + (d - b)**2 - R + r + (w - x)**2 + (z - y)**2

    ans = solve([eq1, eq2, eq3, eq4, eq5], x, y, w, z)
    print(ans)
    # wtt = eval(input("Eq = "))
    # wtc = pi*5**3/16
    # wtr = (b*h**2)/(3+1.8*h/b)

    # wtc = wtc.subs(d, 5.9)
    # b = 3
    # R = solve(36/79 + 36/464 - 1/d,dict=True)
    # symb = list(R[0].keys())[0]
    # print(R, symb, R[0][symb])

    # print(type(66.5) == float64)
    # print("{}".format(R.free_symbols.pop()))
    # print(str(R.args[1]).split(' '))
    # rt = Eq(wt,wtt)
    # rt = solve(rt)
    # wtc = eval('2*b+1')
    # print(wtt, wtc, wtr)
    # e = FA(A=wtc, F=10)
    # print(solve(e - 1))