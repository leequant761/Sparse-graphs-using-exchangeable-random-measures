from collections import Counter

from scipy.special import gamma
from scipy.stats import levy_stable
import numpy as np

def urn_process(W, U_1, U_2):
    """Urn process

    Parameters
    ----------
    W : float
        conditional total length

    U_1 : (list, collections.Counter)
        list is the previous samples of U_(n1)

    U_2 : (list, collections.Counter)
        list is the previous samples of U_(n2)
    """

    eppf(U_1[1], W, sigma)
    return U_1, U_2

def total_eppf(counter, t, sigma):
    """Given rho is stable, EPPF of Poisson-Kingman(rho|t)

    Parameters
    ----------
    counter : collections.Counter
        a counter for previous sample

    t : float

    sigma : float
        conditional eppf considers only stable parameter sigma

    Reference
    ----------
    (Caron, 2015)
    (Pitman, 2003)
    """
    m_list = list(counter.values())
    # new ball ; p
    p = eppf_divide(m_list, t, sigma)
    
    # previous ball ; (1-p)
    

def eppf_divide(m_list, t, sigma):
    """Given rho is stable, EPPF of Poisson-Kingman(rho|t)

    Parameters
    ----------
    m_list : list
        counting values for previous sample

    t : float

    sigma : float
        conditional eppf considers only stable parameter sigma

    Reference
    ----------
    (Caron, 2015)
    (Pitman, 2003)

    Document
    ----------
    ./doc/eppf.pdf
    """
    np.random.seed(100)
    
    n = sum(m_list)
    k = len(m_list)
    
    numerator = sigma * gamma(n-k*sigma) / t
    denominator = gamma(n+1-(k+1)*sigma)

    alpha, beta = sigma, 1.
    stable = levy_stable(alpha, beta)

    s = stable.rvs(10000)
    mc_approx1 = np.where(s < t, (t-s)**(n-(k+1)*sigma), 0.)
    s = stable.rvs(10000)
    mc_approx2 = np.where(s < t, (t-s)**(n-k*sigma-1), 0.)

    p = numerator * mc_approx1.mean() / (denominator * mc_approx2.mean())
    return p

if __name__=='__main__':
    print(eppf_divide([1]*1000, t=300, sigma=0.5))