import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
t=np.linspace(0, 10, 1000)
y0=[2, -3]
x=np.exp(-t) + np.exp(-2*t)
plt.plot(t, x, 'k')
A=np.array([[0, 1], [-2, -3]])
def linear_ode(t, y):
    return A @ y
linear_ode_solution=solve_ivp(linear_ode, (0, 10), y0, t_eval=t)
y=linear_ode_solution.y
plt.plot(t, y[0, :], 'r--')
plt.xlabel("Time [s]")
plt.ylabel('Solution x')
plt.legend(['Analytic', 'RK45'])
plt.grid(True)
plt.show()
eigvals, eigvacs=np.linalg.eig(A)
print(eigvals)