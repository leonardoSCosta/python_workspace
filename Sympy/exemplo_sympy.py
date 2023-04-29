from sympy import Function, Symbol, solve, Sum, log
import sympy

#  x, y = sympy.symbols('x y')
#
#  eq_teste = "4*x + 3 + y"
#
#  print(solve(eq_teste))


i, n, a, b, d = sympy.symbols('i n a b d')

soma = Sum(a**i * n**2, (i, 0, log(n, b))) + d*a**(log(n, b))

eq_teste = soma

print(solve(eq_teste))
exp = eq_teste.evalf(subs={a: 2, b: 2, d: 1})
print(exp.evalf(subs={a: 2, b: 2, d: 1}))
