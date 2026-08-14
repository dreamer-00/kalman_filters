from gh_filter_2 import g_h_filter
import numpy as np
import matplotlib.pyplot as plt
def gen_data(x0, dx, count, noise_factor):
    return [x0+dx*i+np.random.randn()*noise_factor for i in range(count)]
measurements=gen_data(x0=0, dx=1, count=30, noise_factor=1)
filtered_data=g_h_filter(data=measurements, x0=0, dx=1, g=0.2, h=0.02, dt=1)
plt.figure(figsize=(10,5))
plt.plot(measurements, marker='o', color='red', linestyle='-.', label='Noisy Sensor data (z)')
plt.plot(filtered_data, color='green', linestyle='--', label='Filtered estimate (x)')
true_path=[0+1*i for i in range(30)]
plt.plot(true_path, color='blue', linewidth=2, label="True physical path")
plt.title("Stress testing for g-h filter")
plt.xlabel('Time step')
plt.ylabel('Position')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(loc='best')
plt.tight_layout()
plt.show()
zs_ringing=gen_data(x0=5, dx=2, count=100, noise_factor=10)
data_ringing=g_h_filter(data=zs_ringing, x0=100.0, dx=2.0, g=0.2, h=0.01)
zs_noise=gen_data(x0=5, dx=2, count=100, noise_factor=100)
data_noise=g_h_filter(data=zs_noise, x0=5.0, dx=2.0, g=0.2, h=0.02)
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
# Plot A: Ringing
axs[0].plot(zs_ringing, color='red', marker='.', linestyle='', alpha=0.5, label='Measurements (z)')
axs[0].plot(data_ringing, color='blue', linewidth=2, label='Filter Estimate (x)')
axs[0].plot([5 + 2*i for i in range(100)], color='green', linestyle='--', label='True Path')
axs[0].set_title('Scenario A: Bad Initial Guess (Ringing)')
axs[0].set_xlabel('Time Step')
axs[0].set_ylabel('Position')
axs[0].grid(True, linestyle=':', alpha=0.7)
axs[0].legend()
# Plot B: Extreme Noise
axs[1].plot(zs_noise, color='red', marker='.', linestyle='', alpha=0.5, label='Measurements (z)')
axs[1].plot(data_noise, color='blue', linewidth=2, label='Filter Estimate (x)')
axs[1].plot([5 + 2*i for i in range(100)], color='green', linestyle='--', label='True Path')
axs[1].set_title('Scenario B: Extreme Noise')
axs[1].set_xlabel('Time Step')
axs[1].grid(True, linestyle=':', alpha=0.7)
axs[1].legend()

plt.tight_layout()
plt.show()