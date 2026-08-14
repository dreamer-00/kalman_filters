import numpy as np
import matplotlib.pyplot as plt
from gh_filter_2 import g_h_filter
zs=[5, 6, 7, 8, 9, 10, 11, 12, 13, 14]+[14]*20
data_low=g_h_filter(data=zs, x0=4.0, dx=1.0, g=0.1, h=0.01)
data_mid=g_h_filter(data=zs, x0=4.0, dx=1.0, g=0.5, h=0.01)
data_high=g_h_filter(data=zs, x0=4.0, dx=1.0, g=0.9, h=0.01)
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
# Plot Low G
axs[0].plot(zs, marker='o', color='red', linestyle='', label='Measurements')
axs[0].plot(data_low, color='blue', linewidth=2, label='Filter')
axs[0].set_title('Low g (0.1): Massive Overshoot')
axs[0].grid(True, linestyle=':')
# Plot Mid G
axs[1].plot(zs, marker='o', color='red', linestyle='', label='Measurements')
axs[1].plot(data_mid, color='blue', linewidth=2, label='Filter')
axs[1].set_title('Mid g (0.5): Moderate Overshoot')
axs[1].grid(True, linestyle=':')
# Plot High G
axs[2].plot(zs, marker='o', color='red', linestyle='', label='Measurements')
axs[2].plot(data_high, color='blue', linewidth=2, label='Filter')
axs[2].set_title('High g (0.9): Fast Adaptation')
axs[2].grid(True, linestyle=':')
plt.tight_layout()
plt.show()