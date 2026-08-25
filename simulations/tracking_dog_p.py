import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
hallway=np.array([1,1,0,0,0,0,0,0,1,0])
def normalize(p):
    return p/np.sum(p)
def scaled_update(hall, belief, z, z_prob):
    scale=z_prob/(1.0-z_prob)
    belief[hall==z] *=scale
    return normalize(belief)
belief=np.array([0.1]*10)
belief=scaled_update(hallway, belief, z=1, z_prob=0.75)
def bar_plot(belief, title="Probability Distribution", ylim=(0, 0.4)):
    plt.figure(figsize=(9, 3.5))
    positions=np.arange(len(belief))
    plt.bar(positions, belief, width=0.6, color='#3498db', edgecolor='black', alpha=0.85)
    plt.xticks(positions)
    plt.xlabel('Hallway Position')
    plt.ylabel('Probability')
    plt.title(title)
    plt.ylim(ylim)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
print('sum= ', sum(belief))
print('probability of door= ', belief[0])
print('probability of wall= ', belief[2])
bar_plot(belief)