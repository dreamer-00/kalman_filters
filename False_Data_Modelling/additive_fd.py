import numpy as np
import matplotlib.pyplot as plt

# Problem 1
A = np.array([[0.6, 0.4], [0.4, 0.6]])
C = np.array([[1, 1]])
Q = np.array([[0.1, 0], [0, 0.1]])
R = np.array([[0.1]])
P_init = np.array([[4.0, 0.0], [0.0, 4.0]])
X_init = np.array([[2.0], [1.0]])
pa = 0.4
mu_a = 5.0
sigma_a = 2.0**2
runs = 100
time_steps = 300
errors = np.zeros((runs, time_steps, 2))
errors_fd = np.zeros((runs, time_steps, 2))
errors_red=np.zeros((runs, time_steps, 2))

for run in range(runs):
    X_true = X_init.copy()
    init_offset = np.random.multivariate_normal([0, 0], P_init).reshape(2, 1)
    X_est = X_init + init_offset
    X_est_fd = X_init + init_offset
    X_est_red=X_init + init_offset
    P_est = P_init.copy()
    P_est_fd = P_init.copy()
    P_est_red = P_init.copy()

    for k in range(time_steps):
        w = np.random.multivariate_normal([0, 0], Q).reshape(2, 1)
        v = np.random.multivariate_normal([0], R)
        X_true = A @ X_true + w
        Y_normal = C @ X_true + v

        beta_a = np.random.binomial(1, pa)
        a_k = np.random.normal(mu_a, np.sqrt(sigma_a))
        Y_attacked = Y_normal + beta_a * a_k

        # Condition 1: Normal
        X_pred = A @ X_est
        P_pred = A @ P_est @ A.T + Q
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T @ np.linalg.inv(S)
        X_est = X_pred + K @ (Y_normal - C @ X_pred)
        P_est = (np.eye(2) - K @ C) @ P_pred

        # Condition 2: Attacked measurements
        X_pred_fd = A @ X_est_fd
        P_pred_fd = A @ P_est_fd @ A.T + Q
        S_fd = C @ P_pred_fd @ C.T + R
        K_fd = P_pred_fd @ C.T @ np.linalg.inv(S_fd)
        X_est_fd = X_pred_fd + K_fd @ (Y_attacked - C @ X_pred_fd)
        P_est_fd = (np.eye(2) - K_fd @ C) @ P_pred_fd
        
        #Condition 3: Redesigned Filter
        X_pred_red=A @ X_est_red
        P_pred_red=A @ P_est_red @ A.T + Q
        Y_est_red=(C @ X_pred_red)[0,0] + pa * mu_a
        P_yy_red=(C @ P_pred_red @ C.T)[0,0] + R[0,0] + pa * sigma_a + pa * (1-pa) * (mu_a**2)
        P_xy_red= P_pred_red @ C.T
        K_red=P_xy_red / P_yy_red
        innov_red= Y_attacked - Y_est_red
        X_est_red= X_pred_red + K_red * innov_red
        P_est_red = P_pred_red - (K_red @ K_red.T) * P_yy_red
        
        #Store estimation errors
        errors[run, k, 0] = X_true[0, 0] - X_est[0, 0]
        errors[run, k, 1] = X_true[1, 0] - X_est[1, 0]
        errors_fd[run, k, 0] = X_true[0, 0] - X_est_fd[0, 0]
        errors_fd[run, k, 1] = X_true[1, 0] - X_est_fd[1, 0]
        errors_red[run, k, 0] = X_true[0, 0] - X_est_red[0, 0]
        errors_red[run, k, 1] = X_true[1, 0] - X_est_red[1, 0]
#Compute monte carlo rmse across all runs
rmse = np.sqrt(np.mean(errors**2, axis=0))
rmse_fd = np.sqrt(np.mean(errors_fd**2, axis=0))
rmse_red=np.sqrt(np.mean(errors_red**2, axis=0))

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
fig.suptitle('Monte Carlo Analysis: Normal vs Unmodified vs Redesigned KF', fontsize=14)

# State 1 RMSE (Position)
axs[0].plot(rmse[:, 0], label='Normal (No Attack)', color='blue', linewidth=1.5)
axs[0].plot(rmse_fd[:, 0], label='Unmodified KF (Under Attack)', color='red', linestyle='--')
axs[0].plot(rmse_red[:, 0], label='Redesigned KF (Under Attack)', color='green', linewidth=2)
axs[0].set_ylabel('Position RMSE')
axs[0].legend()
axs[0].grid(True, linestyle=':', alpha=0.7)

# State 2 RMSE (Velocity)
axs[1].plot(rmse[:, 1], label='Normal (No Attack)', color='blue', linewidth=1.5)
axs[1].plot(rmse_fd[:, 1], label='Unmodified KF (Under Attack)', color='red', linestyle='--')
axs[1].plot(rmse_red[:, 1], label='Redesigned KF (Under Attack)', color='green', linewidth=2)
axs[1].set_ylabel('Velocity RMSE')
axs[1].set_xlabel('Time Step (k)')
axs[1].legend()
axs[1].grid(True, linestyle=':', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.show()