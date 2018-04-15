import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pylab as plt
RMSE_START = 20

from kfsims.common import init_trajectory, init_all

def shifted_sinus(N, sin_halves=5, shift=1):
    a = np.array(shift + np.sin([np.pi * (sin_halves*i/N) for i in range(N)]) * 5)
    return np.array([a,a])

def rising_sinus(N, sin_halves=5, shift=0):
    np.random.seed(10)
    a = shift + np.sin([np.pi * (sin_halves*i/N) for i in range(N)]) + np.random.rand(N)*2
    return np.array([a,a])

#noise_modifier = shifted_sinus(300)
noise_modifier = rising_sinus(300)

#noise_modifier = shifted_sinus(300)
#noise_modifier = rising_sinus(300)

from kfsims import common
from kfsims.node import node_factory, observe_factory

trj = init_trajectory()
msrm = trj.Y + noise_modifier
true_traj = trj.X.T

def daniels_variant(measurements, true):
    iterations = 10
    _, xk, P, tau, rho, u, U, H, F, Q, N = common.init_all()
    nd = node_factory(xk, P, u, U, F, Q, H, rho, tau, observe_factory(measurements.T), iterations)
    nd()
    preds = np.array(nd.logger['x']).squeeze()
    return preds, nd.post_rmse(true.T, start_element=RMSE_START)
res_dv, rms_dv = daniels_variant(msrm, true_traj)
print(rms_dv)