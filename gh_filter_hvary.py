import numpy as np
import matplotlib.pyplot as plt
from gh_filter_2 import g_h_filter
zs=np.linspace(0, 1, 50)
data_good_guess=g_h_filter(data=zs, x0=0.0, dx=0.0, g=0.2, h=0.05)
data_bad_guess_low=g_h_filter(data=zs, x0=0.0, dx=2.0, g=0.2, h=0.05)
data_bad_guess_high=g_h_filter(data=zs, x0=4.0, dx=2.0, g=0.2, h=0.5)
fig, axs = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
# Plot Low G
axs[0].plot(zs, color='red', label='True signal')
axs[0].plot(data_good_guess, color='blue', linewidth=1, label='Filter')
axs[0].set_title('Good Guess, Low h (0.05)\nSmooth Tracking')
axs[0].grid(True, linestyle=':')
# Plot Mid G
axs[1].plot(zs, color='red')
axs[1].plot(data_bad_guess_low, color='blue', linestyle='--')
axs[1].set_title('Bad Guess (dx=2), Low h (0.05\nMassive, Slow Ringing)')
axs[1].set_ylim([-1, 5])
axs[1].grid(True, linestyle=':')
# Plot High G
axs[2].plot(zs, color='red')
axs[2].plot(data_bad_guess_high, color='blue', linestyle='--')
axs[2].set_title('Bad Guess (dx=2), High h (0.5)\nFast, Tight Ringing')
axs[2].grid(True, linestyle=':')
plt.tight_layout()
plt.show()