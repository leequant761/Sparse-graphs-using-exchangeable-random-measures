import numpy as np
from scipy.stats import poisson
from scipy.special import factorial

def truncate_poisson(w, z):
    """Zero-Truncated Poisson sampler

    Parameters
    ----------
    w : np.array with shape (|N|,)

    z : np.array with shape (|N|, |N|) with binary values

    Reference
    ----------
    (Caron, 2015) (48)
    """
    w = np.array(range(1, 501))

    def truncate_poisson_pmf(lambd:float):
        maximum_support = 1000
        if lambd==0:
            pbt = np.concatenate(([1.], np.zeros(maximum_support)))
        else:
            log_pbt = []
            for k in range(1, maximum_support+1):
                log_p = k*np.log(lambd) - np.log(np.exp(lambd) - 1) - sum([np.log(i) for i in range(1, k+1)])
                log_pbt.append(log_p)
            pbt = np.concatenate(([0.], np.exp(log_pbt)))
            pbt = pbt / sum(pbt) # normalize
        return np.random.choice(range(len(p)), 1, p=pbt)[0]

    tp_pmf = np.vectorize(truncate_poisson_pmf)
    np.random.seed(100)
    # diagonal
    temp = np.diag(w ** 2)
    temp = temp * z.diagonal()
    diagonal_results = tp_pmf(temp)

    # upper triangle
    temp = np.triu(2 * np.outer(w, w) * Z.toarray()[:500, :500], k=1)
    upper_results = tp_pmf(temp)

    return diagonal_results + upper_results + upper_results.T

def update_n(Z, w, n_bar=None):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    Z : np.array or scipy.sparse with shape (|N|, |N|)
        Undirected simple graph

    w : np.array with shape (|N|, )
        sampled atom's weights of CRM W = \sum w_i delta(theta_i)

    n_bar : np.array or scipy.sparse with shape (|N|, |N|)
        If is not None, it will impute the latent variable n_bar using MH algorithm.
    
    Reference
    ----------
    (Caron, 2015) (48) / E.1. Step 3
    """
    if n_bar is None:
        n_bar = truncate_poisson(w=w, z=Z)
    else:
        N_alpha = len(w)
        # define proposal
        prob_table = np.random.uniform(size=(N_alpha, N_alpha))
        one_table = (n_bar == 1)
        over_table = (n_bar > 1)
        one_table = one_table * 1
        over_table = over_table * (2*(prob_table > 0.5) - 1)
        n_bar_tilde = n_bar + one_table + over_table
        
        # compute acceptance probability
        term1 = factorial(n_bar) / factorial(n_bar_tilde)
        # NOTE: I think the below expression is correct
        # term2 = (2 - np.identity(N_alpha, dtype=int)) * np.outer(w, w)
        term2 = (1 + np.identity(N_alpha, dtype=int)) * np.outer(w, w)
        term2 = term2 ** (n_bar_tilde - n_bar)
        term3_n = np.ones((N_alpha, N_alpha))
        term3_n[n_bar_tilde > 1] = 0.5
        term3_d = np.ones((N_alpha, N_alpha))
        term3_d[n_bar > 1] = 0.5
        term3 = term3_n / term3_d
        r = term1 * term2 * term3
        r[r>1] = 1.

        # MH samples
        is_accept = np.random.binomial(1, r)
        n_bar = n_bar * (1-is_accept) + n_bar_tilde * is_accept

    return n_bar