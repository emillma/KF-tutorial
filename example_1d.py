import numpy as np
from scipy.stats import multivariate_normal
from numpy.random import normal
from matplotlib import pyplot as plt
"""
x: state
p: state covariance
q: state update covariance
r: measurement covariance (if we konw mean)
s: measurement covariance (if we use h@x as estimate of mean)
k: kalman gain
"""

# just a renaming of function to get random gaussian value
random_gaussian = multivariate_normal.rvs

# lists to store data
x_gt_seq = []
x_est_seq = []
z_seq = []

p_seq = []
k_seq = []

# parameters
q = 0.1**2
r = 0.4**2

x_k_gt = 1
p_est = 10**2
x_est = 0
for i in range(1000):
    x_k_gt = x_k_gt + random_gaussian(0, q)  # true position
    z = x_k_gt + random_gaussian(0, r)  # measurement

    x_pred = x_est  # predicted state estimate
    p_pred = p_est + q  # predicted estimate covariance

    z_pred = x_pred  # predicted measurement
    innovation = z - z_pred  # innovation

    # Innovation covariance (same as predicted measurement covariance)
    s = p_est + r  # innovation

    k = p_est * (1/s)  # compute the calman gain

    x_est = x_est + k * innovation  # get the updated state
    p_est = (1-k)*p_pred  # get the updated covriance

    x_gt_seq.append(x_k_gt)
    z_seq.append(z)
    k_seq.append(k)
    x_est_seq.append(x_est)
    p_seq.append(p_est)

fig, ax = plt.subplots(4, 1, figsize=(8, 8))
ax[0].plot(x_gt_seq, label='gt')
ax[0].plot(x_est_seq, label='x')
ax[1].plot(z_seq, label='z')
ax[2].plot(p_seq, label='p')
ax[3].plot(k_seq, label='k')

for a in ax:
    a.legend()
plt.show()
