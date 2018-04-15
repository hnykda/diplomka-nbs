import matplotlib.pylab as plt
from scipy.stats import multivariate_normal as mvn
import numpy as np

def plot_variants_only(av, kfc, true, start_pos):
    f, axs = plt.subplots(2, 2, figsize=(15, 10))
    for sl, ax in enumerate(axs.reshape(-1)):
        ax.plot(av[start_pos:, sl], label='VBKF', alpha=0.8)
        ax.plot(kfc[start_pos:, sl], label='Classic KF', alpha=0.8)
        ax.plot(true[start_pos:, sl], label='True', alpha=0.8)
        ax.legend()
        ax.set_title('{}. coord'.format(sl))

def plot_single(ax1, sl, av, kfc, true, measurements, start_pos):
    ax1.plot(av[start_pos:, sl], label='VBKF', alpha=0.8)
    ax1.plot(kfc[start_pos:, sl], label='Classic KF', alpha=0.8)
    ax1.plot(true[start_pos:, sl], label='True', alpha=0.8)
    ax1.plot(measurements.T[start_pos:, sl], label='Measurements', alpha=0.2)
    ax1.legend()

def plot_variants(av, kfc, measurements, true, start_pos):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plot_single(ax1, 0, av, kfc, true, measurements, start_pos=start_pos)
    plot_single(ax2, 1, av, kfc, true, measurements, start_pos=start_pos)
    
def static_noise(N, mod=1, d=2):
    return mvn.rvs(cov=np.eye(d) * mod, size=N)

def hill_noise(N, low=1, mid=10, top=15):
    """  ____
        /
    ___/
    """
    lower = mvn.rvs(cov=np.eye(2)*low, size=20)
    middle = np.array([mvn.rvs(cov=np.eye(2)*i, size=1) for i in range(mid)])
    upper = mvn.rvs(cov=np.eye(2)*top, size=N-mid-20)
    return np.concatenate([lower, middle, upper])

def sin_noise(N, sin_halves=2):
    a = np.sin([np.pi * (sin_halves*i/N) for i in range(N)])
    return np.array([a,a]).T