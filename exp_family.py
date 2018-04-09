import numpy as np

def init_IW_hyp(D, d):
    """Initialize default hyperparameter"""
    return [-0.5 * D, -0.5 * (d + D.shape[0] + 1)]


def get_IW_var_from_hyp(hyp):
    """
    prior_iWishart.psi
    
    Seems OK, although iff `self.natparam[:self.d, :self.d]`
    means the latest "matrix-part" of the hyperparametr
    """
    return -2 * hyp[0]

def get_IW_dof_from_hyp(hyp):
    """
    prior_iWishart.nu
    
    `-2. * self.natparam[-1,-1] - self.d - 1` 
    Seems OK, if `self.natparam[-1,-1]` is DOF
    """
    return -2 * hyp[1] - hyp[0].shape[0] - 1

def get_E_IW_hyp(hyp):
    """
    prior_iWishart.Esigma
    """
    psi = get_IW_var_from_hyp(hyp) 
    nu = get_IW_dof_from_hyp(hyp)
    return psi / (nu - psi.shape[0] - 1)

def get_suff_IW_conj(x, y, pref):
    """
    Gets a sufficient vector for Norm-IW conjugated
    update
    """
    diff = x - y
    suff_D = np.array([
        -1/2 * (pref + np.outer(diff, diff)),
        -1/2,
    ])
    return suff_D

get_sufficient_vector = get_suff_IW_conj

def sum_hyp_and_suff(_hyp, suff):
    """
    Sums both elements of hyperparametr "suff"
    and suff stat.
    """
    return [
        _hyp[0] + suff[0],
        _hyp[1] + suff[1]
    ]

def update_IW(hyp_D_prev, xikk, xk, Pik_old):
    """
    Do an update of Norm-IW conjugate in an exponential form.
    """
    # get's a sufficient statistics
    suff_D = get_suff_IW_conj(xikk, xk, Pik_old)
    # add it to the hyperparametr value
    hyp_D = sum_hyp_and_suff(hyp_D_prev, suff_D)
    # compute new expected value of a scale matrix
    Dik = get_E_IW_hyp(hyp_D)
    return Dik, hyp_D

def get_IW_pars_from_hyp(hyp_R):
    Ukk = get_IW_var_from_hyp(hyp_R)
    ukk = get_IW_dof_from_hyp(hyp_R)
    return Ukk, ukk

def init_P_hyp(tau, P):
    init = np.array([
        -0.5 * tau * P,
        -0.5 * tau - 1 - P.shape[0]
    ])
    return init

def init_R_hyp(rho, psi, nu):
    m = psi.shape[0]
    init = np.array([
            rho * psi,
            -0.5 * ((rho * (-2 * nu - m - 1)) + m + 1)
            ])
    return init
