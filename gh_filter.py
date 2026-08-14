from matplotlib.lines import lineStyles
import numpy as np
import matplotlib.pyplot as plt
measurements=[158.0, 164.2, 160.3, 159.9, 162.1, 164.6, 169.6, 167.4, 166.4, 171.0, 171.2, 172.6]
estimated_weight=160.0
gain_rate=1.0 #initial guess of weight gain (1lb/day)
scale_factor=0.4 #40% trust in scale (g)
gain_factor=0.33 #33% trust in weight gain (h)
time_step=1.0 #1 day
estimates=[]
for z in measurements:
    estimated_weight=estimated_weight+(gain_rate*time_step)
    gain_rate=gain_rate
    residual=z-estimated_weight
    gain_rate=gain_rate+gain_factor*(residual/time_step)
    estimated_weight=estimated_weight+(scale_factor*residual)
    estimates.append(estimated_weight)
days=range(1, len(measurements)+1)
plt.figure(figsize=(10,5))
plt.plot(days, measurements, color='red', marker='o', linestyle='solid', label='Noisy Scale Measurements')
plt.plot(days, estimates, color='blue', marker='o', linestyle='--', label='Filter Estimate')
actual_trend=[160+i for i in range(len(measurements))]
plt.plot(days, actual_trend, color='green',linestyle='-', label='Actual Hidden Trend')
plt.title('Self-contained g-h filter: weight tracking')
plt.xlabel('Days')
plt.ylabel('Weight(lbs)')
plt.xticks(days)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()

