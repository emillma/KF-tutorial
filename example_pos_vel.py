import numpy as np
from numpy.random import normal
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
"""
x: state
P: state covariance
Q: state update covariance
R: measurement covariance (if we konw mean)
S: measurement covariance (if we use h@x as estimate of mean)
K: kalman gain
"""
random_gaussian = multivariate_normal.rvs  # just a renaming

# lists to store data
x_gt_seq = []
x_est_seq = []
z_seq = []

P_seq = []
K_seq = []

dt = 0.05  # timestep

# parameters (the discretization here is first order and can be improved)
F = np.array([[1, dt],
              [-0.5*dt, 1]])

Q = np.array([[0, 0],
              [0, 1**2*dt]])

H = np.array([[1, 0]])
R = np.array([[10**2/dt]])

# initial states
x_gt = np.array([10, 10])

x_est = np.array([0, 0])
P_est = np.diag([1, 1])

for i in range(1000):
    x_gt = F @ x_gt + random_gaussian(np.array([0, 0]), Q)  # true position

    z = H @ x_gt + random_gaussian(0, R)  # measurement

    x_pred = F @ x_est  # predicted state estimate
    P_pred = F @ P_est @ F.T + Q  # predicted estimate covariance

    z_pred = H @ x_pred  # predicted measurement
    innovation = z - z_pred  # innovation

    # Innovation covariance (same as predicted measurement covariance)
    S = H @ P_est @ H.T + R
    K = P_est @ H.T @ np.linalg.inv(S)

    x_est = x_pred + K @ innovation
    P_est = (np.eye(2) - K @ H) @ P_pred

    x_gt_seq.append(x_gt)
    x_est_seq.append(x_est)
    z_seq.append(z)
    P_seq.append(P_est)
    K_seq.append(K)

fig, ax = plt.subplots(5, 1)
ax[0].plot([x[0] for x in x_gt_seq], label='pos_gt')
ax[0].plot([x[0] for x in x_est_seq], label='pos_est')
ax[1].plot([x[1] for x in x_gt_seq], label='vel_gt')
ax[1].plot([x[1] for x in x_est_seq], label='vel_est')

ax[2].plot([z[0] for z in z_seq], label='z')

ax[3].plot([P[0, 0] for P in P_seq], label='p_pos')
ax[3].plot([P[1, 1] for P in P_seq], label='p_vel')
ax[3].plot([P[1, 0] for P in P_seq], label='p_pos_vel')

ax[4].plot([K[0] for K in K_seq], label='k_pos')
ax[4].plot([K[1] for K in K_seq], label='k_vel')

for a in ax:
    a.legend()
plt.show(block=True)
