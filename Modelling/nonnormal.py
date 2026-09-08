import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
y0=[0, 1]
t1=np.arange(0, 1000.1, 0.1)
A1=np.array([[-0.009, 1], [0, -0.01]])
def system1(t, y):
    return A1@y
sol1=solve_ivp(system1, (0, 1000), y0, t_eval=t1)
plt.figure()
plt.plot(sol1.t, sol1.y[0], label='x')
plt.plot(sol1.t, sol1.y[1] ,label='y')
plt.xlabel('Time')
plt.ylabel('x, v')
plt.legend()
plt.grid(True)
plt.show()
t2=np.arange(0, 20.01, 0.01)
A2=np.array([[-1, 1], [0, -1]])
def system2(t, y):
    return A2@y
sol2=solve_ivp(system2, [t2[0], t2[-1]], y0, t_eval=t2)
plt.figure()
plt.plot(sol2.t, sol2.y[0], label='x')
plt.plot(sol2.t, sol2.y[1], label='y')
plt.xlabel('Time')
plt.ylabel('x, v')
plt.legend()
plt.grid(True)
plt.show()