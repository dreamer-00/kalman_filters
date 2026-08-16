import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import measure
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
def compute_new_position(pos, vel, dt=1):
    return pos + (vel*dt)
def measure_position(pos, noise_std=500.0):
    return pos+np.random.randn()*noise_std
def gen_train_data(pos, vel, count, accel=0.0):
    zs=[]
    true_positions=[]
    for _ in range(count):
        pos=compute_new_position(pos, vel)
        vel+=accel
        true_positions.append(pos)
        zs.append(measure_position(pos))
    return np.asarray(zs), np.asarray(true_positions)
start_pos=23000.0 #23km
start_vel=15.0    #15m/s
time_steps=100
zs, true_pos = gen_train_data(pos=start_pos, vel=start_vel, count=time_steps, accel=0.2)
data_lag=g_h_filter(data=zs, x0=start_pos, dx=start_vel, g=0.01, h=0.0001)
data_jitter=g_h_filter(data=zs, x0=start_pos, dx=start_vel, g=0.01, h=0.001)
zs_km=zs/1000.0
true_pos_km=true_pos/1000.0
data_lag_km=np.array(data_lag)/1000.0
data_jitter_km=np.array(data_jitter)/1000.0
time_axis=range(time_steps)
plt.figure(figsize=(12, 6))
plt.plot(time_axis, zs_km, marker='.', color='red', linestyle='', alpha=0.3, label='Noisy Sensor (500m)')
plt.plot(time_axis, true_pos_km, color='green', linewidth=2, label='True Train Position')
plt.plot(time_axis, data_lag_km, color='blue', linewidth=2, label='Filter A (h=0.0001): Smooth but lags')
plt.plot(time_axis, data_jitter_km, color='orange', linewidth=2, label='Filter B (h=0.001): Tracks accel but jitters')
plt.title('Tracking an Accelerating Train (Trade-off between Smoothness and Reactivity)')
plt.xlabel('Time (seconds)')
plt.ylabel('Position (km)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()