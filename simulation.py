from scipy.stats import poisson
import numpy as np

from ets import ets_sampling_Caron
from simulation_utils import urn_process 

def simulation(alpha, sigma, tau):
    """It generates the graph data following 7.1. Simulated data (Caron, 2015)

    Parameters
    ----------
    alpha : float
        It specifies Lebesgue measure's support as [0, alpha]. 
        For detail, see Caron(2015)

    sigma : float in (0, 1)
        It is related to stable distribution's parameter.
        Should be in (0, 1), and for detail, see Caron(2015)

    tau : float in (0, inf)
        It is related to exponential tilting. See Caron(2015)

    n_sample : int
        The number of samples

    Reference
    ----------
    (Caron, 2015)
    """
    alpha=300
    sigma=0.5
    tau=1

    np.random.seed(100)
    W_a_star = ets_sampling_Caron(alpha, sigma, tau, n_sample=1)[0]
    D_a_star = poisson(W_a_star**2).rvs()

    U_1, U_2 = None, None
    for _ in range(D_a_star):
        U_1, U_2 = urn_process(W_a_star, U_1, U_2, sigma)