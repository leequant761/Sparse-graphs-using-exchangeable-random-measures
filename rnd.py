from collections import Counter

from scipy.stats import poisson, gamma, uniform, expon
from scipy.special import gammaln
import numpy as np
from numpy import log as log

def ggprnd(alpha, sigma, tau, trc):
    """It samples W = \sum w_i \delta_{\theta_i} from Generalized Gamma Process

    Parameters
    ----------
    alpha : float
        It specifies Lebesgue measure's support as [0, alpha]. 
        For detail, see Caron(2015)

    sigma : float in (-infty, 1)
        It is related to stable distribution's parameter.
        Should be in (0, 1), and for detail, see Caron(2015)

    tau : float in (0, inf)
        It is related to exponential tilting. See Caron(2015)

    trc : float
        Truncation level; do not sample w_i smaller than this value

    Reference
    ----------
    (Caron, 2015)
    https://github.com/misxenia/SNetOC/GGP/ggprnd.m
    """
    if sigma < -1e-8:
        # total mass means intensity measure of [0,infty) X [0, alpha]
        # total_mass = - alpha * tau**sigma / (sigma)
        total_mass = np.exp(log(alpha) - log(-sigma) + sigma*log(tau))
        num_jumps = poisson(total_mass).rvs()
        w = gamma(-sigma, 1/tau).rvs(num_jumps)
        return w

    # set sigma=0 if in [-1e-8, 0] - This needs to be fixed
    sigma = max(sigma, 0)
    A = 5

    ############################################################################
    # TODO December 11, 2020: I don't know how to derive it yet. I just copied
    ############################################################################
    if sigma > 0:
        # Use a truncated Pareto on [trc, A]
        log_total_mass = log(alpha) - log(sigma) - tau*trc -\
                    gammaln(1-sigma) + log(trc**(-sigma) - A**(-sigma))
        num_jumps = poisson(np.exp(log_total_mass)).rvs()

        # sample from truncated Pareto
        num = -(uniform().rvs(num_jumps) * (A**sigma - trc**sigma) - A**sigma)
        dem = (A*trc)**sigma
        log_w1 = -1/sigma * log(num / dem)
    else:
        log_total_mass = log(alpha) - tau*trc - gammaln(1-sigma) + \
                        log(log(A) - log(trc))
        num_jumps = poisson(np.exp(log_total_mass)).rvs()
        log_w1 = uniform().rvs(num_jumps) * (log(A) - log(trc))  + log(trc)
    w1 = np.exp(log_w1)
    ind1 = log(uniform().rvs(num_jumps)) < tau*(trc - w1)
    w1 = w1[ind1]
    # Use a truncated exponential on (A, infty)
    log_total_mass = log(alpha) - tau*A - (1+sigma)*log(alpha) - log(tau) - gammaln(1-sigma)
    num_jumps = poisson(np.exp(log_total_mass)).rvs()
    log_w2 = log(A + expon(0, 1/tau).rvs(num_jumps))
    ind2 = log(uniform().rvs(num_jumps)) < -(1+sigma) * (log_w2 - log(A))
    w2 = np.exp(log_w2[ind2])

    return np.concatenate([w1, w2])

if __name__=='__main__':
    ggprnd(300, 0.5, 1, 1e-6)