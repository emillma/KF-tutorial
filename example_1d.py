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

    # what is the prediction if we assume a stationary process (f=1, b=0)?
    x_pred = np.nan  # TODO predicted state estimate

    # what is the predicted uncertainty assuming stationary process (f=1)?
    p_pred = np.nan  # TODO predicted estimate covariance

    # what is the predicted measurement assuming h=1?
    z_pred = np.nan  # TODO predicted measurement

    innovation = z - z_pred  # innovation

    # what is the uncertainty of the innovation?
    s = np.nan  # TODO Innovation covariance

    # compute the kalman gain (h=1)
    k = np.nan  # TODO compute the calman gain

    # what are the new estimates?
    x_est = np.nan  # TODO get the updated state
    p_est = np.nan  # TODO get the updated covriance

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
