import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
w=2 * np.pi # natural frequency
d=0.25 
#spring mass damper system
A=np.array([[0, 1], [-w**2, -2*d*w]])
dt=0.10 #time step
T=10 #amount of time to integrate
x0=np.array([[2.0], [0.0]]) #initial conditions (x=2, v=0)
num_steps=int(T/dt)
xF=np.zeros((2, num_steps+1))
xF[:, 0]=x0[:, 0]
tF=np.zeros(num_steps+1)
for k in range(num_steps):
    tF[k+1]=(k+1)*dt
    xF[:, k+1]=(np.eye(2) + dt * A) @ xF[:, k]
t_span=(0, T)
t_eval=np.arange(0, T+dt, dt)
sol=solve_ivp(lambda t, x: A @ x, t_span, x0[:, 0], t_eval=t_eval, method='RK45')
xGood=sol.y.T
# Graph 1: Time vs Position
plt.figure()
plt.plot(tF, xF[0, :], 'k', label='Forward Euler')
plt.plot(sol.t, xGood[:, 0], 'r', label='ODE45 (RK)')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.legend()
plt.show()
# Graph 2: Phase Space (Position vs Velocity)
plt.figure()
plt.plot(xGood[:, 0], xGood[:, 1], 'k')
plt.xlabel('Position [m]')
plt.ylabel('Velocity [m/s]')
plt.show()