import numpy as np
import matplotlib.pyplot as plt
#x=[px, py, v, psi(heading (radians))]
X_true=np.array([0.0, 0.0, 10.0, 0.0])
dt=0.1
acceleration=0.2
yaw_rate=0.05
steps=200
def f(x, a, omega, dt):
    px, py, v, psi=x
    px_new= px + v * np.cos(psi)*dt
    py_new= py + v * np.sin(psi)*dt
    v_new= v + a * dt
    psi_new= psi + omega * dt
    return np.array([px_new, py_new, v_new, psi_new])
print(X_true)
#GNSS measures:
#z=[px, py] + noise
def h(x):
    px=x[0]
    py=x[1]
    return np.array([px, py])
R=np.array([[9.0, 0.0], [0.0, 9.0]])
Q=np.array([[0.05, 0.00, 0.00, 0.00], [0.00, 0.05, 0.00, 0.00], [0.00, 0.00, 0.10, 0.00], [0.00, 0.00, 0.00, 0.01]])
X_est=np.array([0.0, 0.0, 9.0, 0.20])
P=np.array([[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.1]])
L=4
alpha=1.0
beta=2.0
kappa=0.0
lam=alpha**2 * (L+kappa) - L
Wm=np.full(2*L+1, 1/(2*(L+lam)))
Wm[0]=lam/(L+lam)
Wc=Wm.copy()
Wc[0] += 1 - alpha**2 + beta
true_history=[]
gnss_history=[]
ukf_history=[]
for k in range(steps):
    true_process_noise=np.random.multivariate_normal(np.zeros(L), Q)
    X_true=(f(X_true, acceleration, yaw_rate, dt)+true_process_noise)
    measurement_noise=np.random.multivariate_normal(np.zeros(2), R)
    z=h(X_true)+measurement_noise
    S=np.linalg.cholesky((L+lam)*P)
    sigma_points=np.empty((2*L+1, L))
    sigma_points[0]=X_est
    for i in range(L):
        sigma_points[i+1]=(
            X_est + S[:, i]
        )
        sigma_points[i+1+L] = (
            X_est-S[:, i]
        )
    predicted_sigma=np.array([f(sigma, acceleration, yaw_rate, dt) for sigma in sigma_points])
    X_pred=np.sum(Wm[:, None] * predicted_sigma, axis=0)
    P_pred=Q.copy()
    for i in range(2*L+1):
        dx=(predicted_sigma[i]-X_pred)
        P_pred+=(Wc[i] * np.outer(dx, dx))
    measurement_sigma=np.array([h(sigma) for sigma in predicted_sigma])
    z_pred=np.sum(Wm[:, None] * measurement_sigma, axis=0)
    Pzz=R.copy()
    for i in range(2*L+1):
        dz=(measurement_sigma[i]-z_pred)
        Pzz+=(Wc[i]*np.outer(dz, dz))
    Pxz=np.zeros((L, 2))
    for i in range(2*L+1):
        dx=(predicted_sigma[i]-X_pred)
        dz=(measurement_sigma[i]-z_pred)
        Pxz+=(Wc[i]*np.outer(dx, dz))
    K=np.linalg.solve(Pzz, Pxz.T).T
    innovation=z-z_pred
    X_est=(X_pred + K@innovation)
    P=(P_pred-K@Pzz@K.T)
    P=0.5*(P+P.T)
    true_history.append(X_true.copy())
    gnss_history.append(z.copy())
    ukf_history.append(X_est.copy())
true_history=np.array(true_history)
gnss_history=np.array(gnss_history)
ukf_history=np.array(ukf_history)
plt.figure(figsize=(10, 7))
plt.plot(true_history[:, 0], true_history[:, 1], label="True trajectory")
plt.scatter(gnss_history[:, 0], gnss_history[:, 1], s=15, label="GNSS measurements")
plt.plot(ukf_history[:, 0], ukf_history[:, 1], label="UKF estimate")
plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")
plt.title("2D vehicle Navigation - GNSS + UKF")
plt.grid(True)
plt.legend()
plt.axis("equal")
plt.show()


