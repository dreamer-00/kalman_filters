import numpy as np
import matplotlib.pyplot as plt
# Paper: Problem 1
# A New Method for Nonlinear State Estimation Problem
dt=0.01 # sampling interval (s)
b=0.5
d=0.1
Q=b**2 * dt # process noise variance
R=d**2 * dt # measurement noise variance
simulation_time=4.0
steps=int(simulation_time/dt)
time=np.arange(steps) * dt
# Initial conditions from the paper
x0_true=-0.2
x0_est=0.8
P0=2.0
# Nonlinear model
# x_k = x_(k-1)
#       + 5*dt*x_(k-1)*(1 - x_(k-1)^2)
#       + process noise
# Deterministic part:
# f(x) = x + 5*dt*x*(1 - x^2)
def f(x):
    return x+5.0*dt*x*(1.0-x**2)
# Measurement model
# y_k = dt*(x_k - 0.05)^2 + measurement noise
def h(x):
    return dt * (x-0.05)**2
# EKF jacobians
# f(x) = x + 5*dt*x*(1-x^2)
# df/dx = 1 + 5*dt*(1 - 3*x^2)
def F_jacobian(x):
    return 1.0+5.0 * dt * (1.0-3.0 * x**2)
# h(x) = dt*(x-0.05)^2
# dh/dx = 2*dt*(x-0.05)
def H_jacobian(x):
    return 2.0 * dt * (x-0.05)
# UKF parameters
L=1
alpha=1.0
beta=2.0
kappa=0.0
lam=alpha**2 * (L+kappa) - L
# UKF weights
Wm=np.full(2*L+1, 1.0/(2.0*(L+lam)))
Wm[0]=lam/(L+lam)
Wc=Wm.copy()
Wc[0]+=1.0-alpha**2+beta
# EKF function
def run_ekf(measurements):
    x=x0_est
    P=P0
    estimates=np.zeros(steps)
    covariances=np.zeros(steps)
    for k in range(steps):
        x_pred=f(x)
        F=F_jacobian(x)
        P_pred=F*P*F+Q
        y_pred=h(x_pred)
        H=H_jacobian(x_pred)
        S=H*P_pred*H+R
        K=P_pred*H/S
        innovation=measurements[k]-y_pred
        x=x_pred+K*innovation
        P=P_pred-K*S*K
        estimates[k]=x
        covariances[k]=P
    return estimates, covariances
# UKF function
def run_ukf(measurements):
    x=x0_est
    P=P0
    estimates=np.zeros(steps)
    covariances=np.zeros(steps)
    for k in range(steps):
        sigma_points=np.array([x, x+np.sqrt((L+lam)*P), x-np.sqrt((L+lam)*P)])
        predicted_sigma=np.array([f(sigma) for sigma in sigma_points])
        x_pred=np.sum(Wm*predicted_sigma)
        P_pred=Q
        for i in range(2*L+1):
            dx=predicted_sigma[i]-x_pred
            P_pred+=Wc[i]*dx**2
        measurement_sigma=np.array([h(sigma) for sigma in predicted_sigma])
        y_pred=np.sum(Wm*measurement_sigma)
        S=R
        for i in range(2*L+1):
            dy=measurement_sigma[i]-y_pred
            S+=Wc[i]*dy**2
        Pxy=0.0
        for i in range(2*L+1):
            dx=predicted_sigma[i]-x_pred
            dy=measurement_sigma[i]-y_pred
            Pxy+=Wc[i]*dx*dy
        K=Pxy/S
        innovation=measurements[k]-y_pred
        x=x_pred+K*innovation
        P=P_pred-K*S*K
        estimates[k]=x
        covariances[k]=P
    return estimates, covariances
# Generating One true trajectory and measurements
def generate_data():
    x_true=x0_true
    true_states=np.zeros(steps)
    measurements=np.zeros(steps)
    for k in range(steps):
        eta=np.random.normal(0.0, np.sqrt(Q))
        x_true=f(x_true)+eta
        nu=np.random.normal(0.0, np.sqrt(R))
        y=h(x_true)+nu
        true_states[k]=x_true
        measurements[k]=y
    return true_states, measurements
# Monte carlo simulation
M=1000
ekf_squared_errors=np.zeros(steps)
ukf_squared_errors=np.zeros(steps)
for run in range(M):
    true_states, measurements=generate_data()
    ekf_estimates, _ = run_ekf(measurements)
    ukf_estimates, _ = run_ukf(measurements)
    ekf_squared_errors+=(true_states-ekf_estimates)**2
    ukf_squared_errors+=(true_states-ukf_estimates)**2
ekf_rmse=np.sqrt(ekf_squared_errors/M)
ukf_rmse=np.sqrt(ukf_squared_errors/M)
# RMSE Plot
plt.figure(figsize=(10,6))
plt.plot(time, ekf_rmse, label="EKF")
plt.plot(time, ukf_rmse, label="UKF")
plt.xlabel("Time (s)")
plt.ylabel("RMSE")
plt.title("Problem 1: EKF vs UKF RMSE")
plt.grid(True)
plt.legend()
plt.show()
average_ekf_rmse=np.mean(ekf_rmse)
average_ukf_rmse=np.mean(ukf_rmse)
print("Monte Carlo Results")
print(f"Number of runs: {M}")
print(f"EKF average RMSE: {average_ekf_rmse:.4f}")
print(f"UKF average RMSE: {average_ukf_rmse:.4f}")
