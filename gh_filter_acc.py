from turtle import color
import numpy as np
import matplotlib.pyplot as plt
def g_h_filter(data, x0, dx, g, h, dt=1):
    x=x0
    results=[]
    for z in data:
        x_est=x+(dx*dt)
        residual=z-x_est
        dx=dx+h*(residual)/dt
        x=x_est+g*residual
        results.append(x)
    return results
def gen_data(x0, dx, count, noise_factor, accel=0):
    zs=[]
    for i in range(count):
        #generating reading
        zs.append(x0+dx*i+np.random.randn()*noise_factor)
        #increasing the velocity for the next step
        dx+=accel
    return zs
zs=gen_data(x0=10, dx=0, count=20, noise_factor=0, accel=2)
data=g_h_filter(data=zs, x0=10.0, dx=0.0, g=0.2, h=0.02)
plt.figure(figsize=(10,5))
plt.plot(zs, marker='s', color='blue', linestyle='-', label='Measurements (Accelerating)')
plt.plot(data, marker='s', color='blue', linestyle='--', label='Filter Estimate (constant velocity)')
for i in range(2, 20, 3):
    plt.plot([i, i], [data[i], zs[i]], color='gray', linestyle=':')
    if i==14:
        plt.text(i+0.5, (data[i]+zs[i])/2, 'Systemic\nLag Error', color='gray', va='center')
plt.title('The effect of acceleration on g-h filter')
plt.xlabel('Time step')
plt.ylabel('Position')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()