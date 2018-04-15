import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pylab as plt
RMSE_START = 20

from kfsims.common import init_trajectory, init_all

#noise_modifier = shifted_sinus(300)
#noise_modifier = rising_sinus(300)

from kfsims import common
from kfsims.node import node_factory, observe_factory

def shifted_sinus(N, sin_halves=5, shift=1):
    a = np.array(shift + np.sin([np.pi * (sin_halves*i/N) for i in range(N)]) * 5)
    return np.array([a,a])

def rising_sinus(N, sin_halves=5, shift=0):
    np.random.seed(10)
    a = shift + np.sin([np.pi * (sin_halves*i/N) for i in range(N)]) + np.random.rand(N)*2
    return np.array([a,a])

#noise_modifier = shifted_sinus(300)
noise_modifier = rising_sinus(300)

trj = init_trajectory()
measurements = trj.Y + noise_modifier
true_traj = trj.X.T



def predict_state(F_l, x_l__l):
    """
    Step 1
    """
    return F_l @ x_l__l


def predict_PECM(F_l, P_l__l, Q_l):
    """
    Step 2
    """
    return F_l @ P_l__l @ F_l.T + Q_l

def init(P_k__l, tau, n, rho, u_l__l, m, U_l__l):
    """
    Step 3.
    """
    t_k__l = n + tau + 1
    T_k__l = tau * P_k__l
    u_k__l = rho * (u_l__l - m - 1) + m + 1
    U_k__l = rho * U_l__l
    return t_k__l, T_k__l, u_k__l, U_k__l



_, xk, P, tau, rho, u, U, H, F, Q, N = init_all()




n = P.shape[0]
m = U.shape[0]
x_log = []
P_log = []
for _ix, zk in enumerate(measurements.T):
#    print('===== Step: ', t, '======')
    #t += 1

    #### Time update
    xk = predict_state(F, xk)
    P = predict_PECM(F, P, Q)

    #### Measurement update
    # Initialization - step 3
    tkk, Tkk, ukk, Ukk = init(P, tau, n, rho, u, m, U)

    ## VB iters
    Pik = P
    xikk = xk
    for i in range(N):
        # steps 4, 5
        err = np.atleast_2d(xikk - xk)
        Aik = Pik + np.outer(err, err)
        tik = tkk + 1
        Tik = Aik + Tkk

        # steps 6, 7
        err = np.atleast_2d(zk - H.dot(xikk))
        Bik = err.T.dot(err) + H.dot(Pik).dot(H.T)
        uik = ukk + 1
        Uik = Bik + Ukk

        # steps 8
        ERkinv = (uik - m - 1) * np.linalg.inv(Uik)
        EPkinv = (tik - n - 1) * np.linalg.inv(Tik)

        # steps 9
        Pik = np.linalg.inv(EPkinv)
        Rik = np.linalg.inv(ERkinv)

        # steps 10-12
        bracket = H.dot(Pik).dot(H.T) + Rik
        Kik = Pik.dot(H.T).dot(np.linalg.inv(bracket))
        xikk = xk + Kik.dot(zk - H.dot(xk))
        Pik = Pik - Kik.dot(H).dot(Pik)

    xk = xikk
    P = Pik
    u = uik
    U = Uik
    x_log.append(xk)
    P_log.append(P)

x_log = np.array(x_log).squeeze().T
P_log = np.array(P_log).squeeze()

res_av, rms_av = np.array(x_log).squeeze(), np.mean(np.sqrt(((x_log[:, RMSE_START:] - true_traj.T[:, RMSE_START:])**2)), axis=1)
print(rms_av)