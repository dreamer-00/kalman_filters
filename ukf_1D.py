#Estimate a hidden 1D state x from a noisy measurement.
# Initial belief:
#     x ~ N(mean=2.0, P=0.25)

# Nonlinear dynamics:
#     x_next = f(x) + process_noise
#     f(x) = x + 0.1*x^2
#     Q = 0.01

# Measurement:
#     y = h(x) + measurement_noise
#     h(x) = x
#     R = 0.04

# Measurement received:
#     y = 2.3

# UKF parameters (chosen for simple 1D demonstration):
#     L = 1
#     alpha = 1
#     kappa = 0
#     beta = 2

# Since L=1:
#     number of sigma points = 2L + 1 = 3

# Goal of the filter:
#     1. Create sigma points
#     2. Propagate them through the nonlinear function
#     3. Calculate predicted mean and covariance
#     4. Use the measurement to update the estimate
import numpy as np
import matplotlib.pyplot as plt
#Initial Belief
mean=2.0
P=0.25
Q=0.01 #process noise variance
R=0.04 #measurement noise variance
#UKF parameters
L=1
alpha=1.0
kappa=0.0
beta=2.0
lam=alpha**2 * (L+kappa)-L
#sigma-point weights
Wm=np.array([lam/(L+lam), 1/2*(L+lam), 1/2*(L+lam)])
Wc=np.array([lam/(L+lam)+(1-alpha**2+beta), 1/(2*(L+lam)), 1/(2*(L+lam))])
#create sigma points
sigma_points=np.array([mean, mean+np.sqrt((L+lam)*P), mean-np.sqrt((L+lam)*P)])
print("Sigma Points:")
print(sigma_points)
#Non linear state transition
def f(x):
    return x+0.1 * x**2
predicted_sigma=np.array([f(x) for x in sigma_points])
print("\nPredicted Sigma Points: ")
print(predicted_sigma)
#predicted mean
predicted_mean=np.sum(Wm*predicted_sigma)
print("Predicted Mean: ")
print(predicted_mean)
#predicted covariance
diff=predicted_sigma-predicted_mean
predicted_P=np.sum(Wc*diff**2)
predicted_P+=Q
print("\nPredicted covariance: ")
print(predicted_P)
#Measurement Function
def h(x):
    return x
measurement_sigma=np.array([h(x) for x in predicted_sigma])
#Predicted Measurement
predicted_measurement=np.sum(Wm * measurement_sigma)
print("\nPredicted Measurement")
#Measurement Covariance
measurement_diff=(measurement_sigma-predicted_measurement)
Pyy=np.sum(Wc*measurement_diff**2)
Pyy+=R
print("\nMeasurement Covariance:")
print(Pyy)
#Cross Covariance
state_diff=predicted_sigma-predicted_mean
Pxy=np.sum(Wc*state_diff*measurement_diff)
print("\nState-measurement covariance:")
print(Pxy)
#Kalman Gain
k=Pxy/Pyy
print("\nKalman gain:")
print(k)
#Actual Measurement
z=2.3
innovation=z-predicted_measurement
print(innovation)
#Update State
updated_mean=(predicted_mean+k*innovation)
print("\nUpdated mean:")
print(updated_mean)
#Update covariance
updated_P=(predicted_P-k*Pyy*k)
print("\nUpdated covariance:")
print(updated_P)
#plotting the graph
x = np.linspace(1, 3.2, 400)
y = f(x)
plt.figure(figsize=(10, 6))
plt.plot(
    x,
    y,
    label=r"$f(x)=x+0.1x^2$"
)
plt.scatter(
    sigma_points,
    predicted_sigma,
    s=100,
    label="Sigma points"
)
plt.axvline(
    mean,
    linestyle="--",
    label=f"Initial mean = {mean}"
)

plt.axhline(
    predicted_mean,
    linestyle="--",
    label=f"Predicted mean = {predicted_mean:.3f}"
)

for x_i, y_i in zip(sigma_points, predicted_sigma):
    plt.annotate(
        f"({x_i:.1f}, {y_i:.3f})",
        (x_i, y_i),
        xytext=(8, 8),
        textcoords="offset points"
    )
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("1D UKF — Sigma Points Through Nonlinear Dynamics")
plt.grid(True)
plt.legend()
plt.show()