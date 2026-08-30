from turtle import update
import numpy as np
import matplotlib.pyplot as plt
#state=[position, velocity]
mean=np.array([10.0, 2.0])
P=np.array([[1.0, 0.3], [0.3, 0.5]])
print("Mean:")
print(mean)
print("\nCovariance:")
print(P)
#Adding UKF Parameters
L=2 #State of the problem
alpha=1.0 
kappa=0.0
beta=2.0
R=0.5
lam=alpha**2 * (L+kappa) - L
print("\nLambda:")
print(lam)
S=np.linalg.cholesky((L+lam)*P)
print("\nCholesky matrix:")
print(S)
sigma_points=np.array([mean, mean+S[:, 0], mean+S[:, 1], mean-S[:, 0], mean-S[:, 1]])
print("\nSigma Points:")
print(sigma_points)
#Our Model is pk+1 = pk + vk, vk+1 = vk + 0.1vk^2
def f(x):
    p=x[0]
    v=x[1]
    new_p=p+v
    new_v=v+0.1*v**2
    return np.array([new_p, new_v])
predicted_sigma=np.array([f(x) for x in sigma_points])
print("\nPredicted Sigma Points:")
print(predicted_sigma)
Wm=np.array([0.0, 0.25, 0.25, 0.25, 0.25])
predicted_mean=np.sum(Wm[:, None] * predicted_sigma, axis=0)
print("\nPredicted Mean:")
print(predicted_mean)
Wc=np.array([2.0, 0.25, 0.25, 0.25, 0.25])
diff=predicted_sigma-predicted_mean
predicted_p=np.zeros((2,2))
for i in range(5):
    predicted_p += Wc[i] * np.outer(diff[i], diff[i])
Q=np.array([[0.01, 0.0], [0.0, 0.02]])
predicted_p += Q
print("\nPredicted Covariance:")
print(predicted_p)
def h(x):
    p= x[0]
    v= x[1]
    return p**2 + v
measurement_sigma=np.array([h(x) for x in predicted_sigma])
print("\nMeasurement sigma points:")
print(measurement_sigma)
predicted_measurement=np.sum(Wm*measurement_sigma)
print("\nPredicted Measurement:")
print(predicted_measurement)
z=104.0
innovation=z-predicted_measurement
print("\Innovation:")
print(innovation)
measurement_diff=(measurement_sigma-predicted_measurement)
Pzz=np.sum(Wc * measurement_diff**2)
Pzz += R
print("\nMeasurement Covariance:")
print(Pzz)
state_diff=predicted_sigma-predicted_mean
Pxz=np.sum(Wc[:, None] * state_diff * measurement_diff[:, None], axis=0)
print("\nCross Covariance:")
print(Pxz)
K= Pxz/Pzz
print("\nKalman gain:")
print(K)
updated_mean=(predicted_mean + K * innovation)
print("\nUpdated state:")
print(updated_mean)
updated_p=(predicted_p-np.outer(K, K) * Pzz)
print("\nUpdated covariance:")
print(updated_p)
#===============================================
# VISUALIZATION - 2D UKF
#===============================================
plt.figure(figsize=(8, 6))
plt.scatter(sigma_points[:, 0], sigma_points[:, 1], label="Initial sigma points")
plt.scatter(predicted_sigma[:, 0], predicted_sigma[:, 1], label="Predicted sigma points")
plt.scatter(mean[0], mean[1], marker="x", s=100, label="Initial Mean")
plt.scatter(predicted_mean[0], predicted_mean[1], marker="x", s=100, label="Predicted Mean")
plt.scatter(updated_mean[0], updated_mean[1], marker="*", s=150, label="Updated Mean")
for i in range(len(sigma_points)):
    plt.plot([sigma_points[i,0], predicted_sigma[i,0]], [sigma_points[i,1], predicted_sigma[i,1]], linestyle=":")
plt.xlabel("Position")
plt.ylabel("Velocity")
plt.title("2D UKF")
plt.grid(True)
plt.legend
plt.show()