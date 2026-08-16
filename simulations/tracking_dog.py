import numpy as np
import matplotlib.pyplot as plt
hallway=np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
n=len(hallway)
belief=np.array([1.0/n]*n)
def normalize(p):
    return p/sum(p)
def update(hall, prior, z, z_prob):
    scale=z_prob/(1.0-z_prob)
    likelihood=np.ones(len(hall))
    likelihood[hall==z]*=scale
    return normalize(likelihood*prior)
def predict(prior, move):
    result=np.zeros(len(prior))
    for i in range(len(prior)):
        result[i]=prior[(i-move)%len(prior)]
    return result
belief=update(hall=hallway, prior=belief, z=1, z_prob=0.75)
posterior_after_sense=belief.copy()
belief=predict(prior=belief, move=1)
posterior_after_move=belief.copy()
fig, axs =plt.subplots(1, 2, figsize=(12, 4), sharey=True)
axs[0].bar(range(n), posterior_after_sense, color='blue', alpha=0.7)
axs[0].set_title('1. Posterior After Reading Door')
axs[0].set_xticks(range(n))
axs[0].set_xlabel('Hallway Position (0-9)')
axs[0].set_ylabel('Probability')
axs[1].bar(range(n), posterior_after_move, color='green', alpha=0.7)
axs[1].set_title('2. Posterior After Moving Right 1 Space')
axs[1].set_xticks(range(n))
axs[1].set_xlabel('Hallway Position (0-9)')
plt.tight_layout()
plt.show()
