import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

lambda1 = np.pi**2 * 1e-4
lambda2 = 2 * np.pi * 1e-2 

k_m = 0
c_m = 3e4 # B da lista
c = c_m
m_m = 58000e3

beta0 = k_m/m_m
beta1 = c_m/m_m

def setpoint(t):
    if (t % 3600) <= 1800:
        return 20
    else:
        return 0
    
def mass(t):
    M_max = 156000e3
    M_min = 58000e3
    # return (M_min - M_max)/(12 * 3600) * t + M_max
    return M_min

def dxdt(input, t):

    a1, a2, y, dydt, y_m, dydt_m = input

    sp = setpoint(t)
    M = mass(t)

    eta = np.array([[1e9, 0],
                    [0, 1e9]])
    b = np.array([[0], [1]])
    P = np.array([[1e7, 0],
                  [0, 1e7]])

    # Erro
    e = y - y_m
    dedt = dydt - dydt_m

    x = np.array([[e],
                  [dedt]])
    
    # Modelo de referência
    d2ydt_m = lambda2 * (sp - y_m) - lambda1 * dydt_m
    #d2ydt_m = (sp - k_m * y_m - c_m * dydt_m)/m_m

    z = d2ydt_m - beta1 * dedt - beta0 * e
    v = np.array([[z, dydt]]).T

    # Lei de adaptacao
    dadt = -eta @ v @ b.T @ P @ x

    # Malha fechada
    d2ydt = 1/M * (a2 * z + a1 * dydt - c * dydt)

    #d2ydt = (sp - a2*beta0*y - (c + a2*beta1 - c)*dydt) * a2/(M*m_m)

    return [dadt[0, 0], dadt[1, 0], dydt, d2ydt, dydt_m, d2ydt_m]

# Initial condition
x0 = [0, 0, 0, 0, 0, 0]

# Generate time values
t = np.linspace(0, 12 * 3600, 100000)

solution = odeint(dxdt, x0, t)

a1      = solution[:, 0]
a2      = solution[:, 1]
y       = solution[:, 2]
dydt    = solution[:, 3]
y_m     = solution[:, 4]
dydt_m  = solution[:, 5]

fig, axes = plt.subplots(4, 1, figsize=(10, 40))

# axes[0].set_title("Amortecimento estimado ($\hat{a}_1$)")
# axes[0].plot(t, a1)
# axes[0].set_ylabel("Amortecimento (Ns/m)")
# axes[0].set_xlabel("Tempo (s)")

axes[0].set_title("Massa estimada ($\hat{a}_2$)")
axes[0].plot(t, a1)
axes[0].set_ylabel("Massa (kg)")
axes[0].set_xlabel("Tempo (s)")

axes[1].set_title("Resposta em posição")
axes[1].plot(t, y, label="Planta")
axes[1].plot(t, y_m, label="Modelo de referência")
axes[1].set_ylabel("Distância (m)")
axes[1].set_xlabel("Tempo (s)")
axes[1].legend()

axes[2].set_title("Resposta em velocidade")
axes[2].plot(t, dydt, label="Planta")
axes[2].plot(t, dydt_m, label="Modelo de referência")
axes[2].set_ylabel("Velocidade (m/s)")
axes[2].set_xlabel("Tempo (s)")
axes[2].legend()

sp_vect = np.vectorize(setpoint)
axes[3].set_title("Setpoint")
axes[3].plot(t, sp_vect(t))
axes[3].set_ylabel("Distância (m)")
axes[3].set_xlabel("Tempo (s)")

plt.show()
