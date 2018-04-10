import numpy as np
from kfsims.common import init_all, init_trajectory
from kfsims.node import node_factory, observe_factory


def test_single_node():
    traj_ = init_trajectory(5)
    traj, xk, P, tau, rho, u, U, H, F, Q, N = init_all(traj_)
    f = observe_factory(traj)
    m1 = node_factory(xk, P, u, U, F, Q, H, rho, tau, f)

    m1()

    exp_x = np.array([
        [-2.76740e-01, 1.09767e+00, 9.62620e-01, 1.09898e+00],
        [-1.49130e-01, 2.92570e-01, 1.12011e+00, -3.49807e+00],
        [1.45799e+00, 9.18870e-01, 8.70549e+00, 1.40484e+00],
        [8.60280e-01, 6.99180e-01, 2.80428e+00, 3.62000e-03],
        [9.53670e-01, -2.63710e-01, 2.23199e+00, -3.16430e+00]
    ])
    exp_last_P = np.array([[0.53926178, 0.03091239, 1.74759438, 0.0822738],
                           [0.03091239, 0.54178761, 0.08098487, 1.78451349],
                           [1.74759438, 0.08098487, 9.21782513, 0.24729238],
                           [0.0822738, 1.78451349, 0.24729238, 9.44313789]])

    np.testing.assert_allclose(exp_x, np.array(m1.logger['x']), atol=1e-5)
    np.testing.assert_allclose(exp_last_P, m1.logger['P'][-1], atol=1e-5)


def test_single_node_rmse():
    traj, xk, P, tau, rho, u, U, H, F, Q, N = init_all()
    f = observe_factory(traj)
    m1 = node_factory(xk, P, u, U, F, Q, H, rho, tau, f)

    m1()

    x_log = np.array(m1.logger['x']).squeeze().T
    out_rmse_kf = np.sqrt(((x_log[:2] - traj.X[:2]) ** 2).mean())
    out_rmse_nokf = np.sqrt(((traj.Y[:2] - traj.X[:2]) ** 2).mean())

    exp_rmse_kf = 0.4621395030767436
    exp_rmse_nokf = 0.7668096325216082

    np.testing.assert_almost_equal(out_rmse_kf, exp_rmse_kf)
    np.testing.assert_almost_equal(out_rmse_nokf, exp_rmse_nokf)

def test_two_mean():
    traj, xk, P, tau, rho, u, U, H, F, Q, N = init_all()

    # observe_factory is now irrelevant
    m1 = node_factory(xk, P, u, U, F, Q, H, rho, tau, observe_factory(traj))
    m2 = node_factory(xk, P, u, U, F, Q, H, rho, tau, observe_factory(traj))

    zs = traj.Y.T
    zs_dist = traj.Y.T + np.random.normal(size=traj.Y.T.shape) * 0.1

    for z1, z2 in zip(zs, zs_dist):
        x1, P1, hyp_P1, hypR1 = m1.single_kf(z1)
        x2, P2, hyp_P2, hypR2 = m2.single_kf(z2)

    x_log = ((np.array(m1.logger['x']) + np.array(m2.logger['x']))/2).squeeze().T
    out = np.sqrt(((x_log[:2] - traj.X[:2]) ** 2 ).mean()), np.sqrt(((traj.Y[:2] - traj.X[:2]) ** 2 ).mean())
    expected = (0.46242291667779106, 0.7668096325216082)
    np.testing.assert_allclose(out, expected, atol=1e-3)