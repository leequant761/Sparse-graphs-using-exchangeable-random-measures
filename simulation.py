from scipy.stats import poisson, uniform
from scipy.sparse import coo_matrix
import numpy as np

from rnd import ggprnd 

def histc(x, binranges):
  indices = np.searchsorted(binranges, x)
  return np.mod(indices-1, len(binranges)-1)

def GGPgraphrnd(alpha, sigma, tau):
    """It generates the graph data following 7.1. Simulated data (Caron, 2015)

    Parameters
    ----------
    alpha : float
        It specifies Lebesgue measure's support as [0, alpha]. 
        For detail, see Caron(2015)

    sigma : float in (-infty, 1)
        It is related to stable distribution's parameter.
        Should be in (-infty, 0] or (0, 1), and for detail, see Caron(2015)

    tau : float in (0, inf)
        It is related to exponential tilting. See Caron(2015)

    n_sample : int
        The number of samples

    Reference
    ----------
    (Caron, 2015)
    https://github.com/misxenia/SNetOC/GGP/GGPgraphrnd.m
    """
    # `epsilon` truncated sampling
    np.random.seed(100)
    epsilon = 1e-6
    W = ggprnd(alpha, sigma, tau, trc=epsilon)

    W_star = sum(W)
    D_star = poisson(W_star**2).rvs()

    U = W_star * uniform().rvs((D_star, 2))
    
    W_interval = np.concatenate([np.array([0.]), W.cumsum()])

    interval_ranks = histc(U.flatten(), W_interval)
    selected_atom = np.array([False] * len(W))
    selected_atom[np.unique(interval_ranks)] = True
    w_rem = sum(W[~selected_atom])
    w = sum(W[selected_atom])

    # D: directed multi-graph
    hash_table = {key: value for key, value in zip(np.unique(interval_ranks), range(len(np.unique(interval_ranks))))}
    indexer = lambda x: hash_table[x]
    indexer = np.vectorize(indexer)
    D = interval_ranks.reshape(D_star, 2)
    D = coo_matrix((np.ones(D_star), (indexer(D[:, 0]), indexer(D[:, 1]))), 
                    shape=(sum(selected_atom), sum(selected_atom)))
    Z = (D + D.T).astype(bool)

    return coo_matrix(Z), w, w_rem

if __name__=='__main__':
    # Simulation input
    alpha=300
    sigma=0.5
    tau=1
    Z, w, w_rem = GGPgraphrnd(alpha, sigma, tau)
    print(f'Node size is {Z.shape[0]}')
    print(f'The number of edges is roughly {Z.sum() / 2}')