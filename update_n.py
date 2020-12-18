import random
import numpy as np
from scipy.stats import poisson
from scipy.special import factorial
from functools import lru_cache

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

def compute_acceptance(n, r, c, w):
    """It computes the array of acceptance rates

    Parameters
    ----------
    n : np.array with shape (|E|, )
        It is a latent variable n_bar and corresponding to `scipy.sparse.data`

    r : np.array with shape (|E|, )
        It is row indices corresponding to `scipy.sparse.coo_matrix.row`

    c : np.array with shape (|E|, )
        It is column indices corresponding to `scipy.sparse.coo_matrix.col`

    w : np.array with shape (|N|, )
        It is sociability array for each atoms
    """
    prev_one = (n == 1)
    prev_two = (n == 2)
    prev_other = (n > 2)
    q = np.zeros(len(n), dtype=bool)
    q[~prev_one] = np.random.binomial(1, 0.5, sum(~prev_one)).astype(bool)
    #
    # Case1 : n_bar = 1
    #
    case1 = prev_one
    row_idx, col_idx, n_bar = r[case1], c[case1], n[case1]
    n_tilde1 = n_bar + 1
    r1 = 1/2 * ((1+(row_idx!=col_idx)) * w[row_idx] * w[col_idx]) * 1/2
    r1[r1>1] = 1.

    #
    # Case2-1 : n_bar = 2 & n_tilde = 3
    #
    case21 = prev_two & q
    row_idx, col_idx, n_bar = r[case21], c[case21], n[case21]
    n_tilde21 = n_bar + 1
    r21 = 1/(n_tilde21) * ((1+(row_idx!=col_idx)) * w[row_idx] * w[col_idx])
    r21[r21>1] = 1.

    #
    # Case2-2 : n_bar = 2 & n_tilde = 1
    #
    case22 = prev_two & ~q
    row_idx, col_idx, n_bar = r[case22], c[case22], n[case22]
    n_tilde22 = n_bar - 1
    r22 = n_bar / ((1+(row_idx!=col_idx)) * w[row_idx] * w[col_idx]) * 2
    r22[r22>1] = 1.

    #
    # Case3-1 : n_bar > 2 & n_tilde = n_bar + 1
    #
    case31 = prev_other & q
    row_idx, col_idx, n_bar = r[case31], c[case31], n[case31]
    n_tilde31 = n_bar + 1
    r31 = 1/(n_tilde31) * ((1+(row_idx!=col_idx)) * w[row_idx] * w[col_idx])
    r31[r31>1] = 1.

    #
    # Case3-2 : n_bar > 2 & n_tilde = n_bar - 1
    #
    case32 = prev_two & ~q
    row_idx, col_idx, n_bar = r[case32], c[case32], n[case32]
    n_tilde32 = n_bar - 1
    r32 = n_bar / ((1+(row_idx!=col_idx)) * w[row_idx] * w[col_idx])
    r32[r32>1] = 1.

    acceptance_pbt = np.zeros(len(n))
    acceptance_pbt[case1] = r1
    acceptance_pbt[case21] = r21
    acceptance_pbt[case22] = r22
    acceptance_pbt[case31] = r31
    acceptance_pbt[case32] = r32

    n_tilde = np.zeros(len(n))
    n_tilde[case1] = n_tilde1
    n_tilde[case21] = n_tilde21
    n_tilde[case22] = n_tilde22
    n_tilde[case31] = n_tilde31
    n_tilde[case32] = n_tilde32

    return acceptance_pbt, n_tilde


def update_n(Z, w, n_bar=None, proposal_idx=None, sparse=True):
    """Helper function to create a meta-dataset for the Omniglot dataset.

    Parameters
    ----------
    Z : np.array or scipy.sparse with shape (|N|, |N|)
        Undirected simple graph

    w : np.array with shape (|N|, )
        sampled atom's weights of CRM W = \sum w_i delta(theta_i)

    n_bar : np.array with shape (|E|,)
        If is not None, it will impute the latent variable n_bar using MH algorithm.

    sparse : if edge is not sparse, vectorized computation is effective
    
    Reference
    ----------
    (Caron, 2015) (48) / E.1. Step 3
    """
    if n_bar is None:
        n_bar = truncate_poisson(w=w, z=Z)
    else:
        N_alpha = len(w)
        if sparse:
            r_idx, c_idx = proposal_idx[0], proposal_idx[1]
            acceptance_pbt, n_tilde = compute_acceptance(n_bar, r_idx, c_idx, w)
            U = np.random.uniform(0, 1, len(n_tilde))
            n_bar[acceptance_pbt > U] = n_tilde[acceptance_pbt > U]
        else:
            # define proposal
            prob_table = np.random.uniform(size=(N_alpha, N_alpha))
            one_table = (n_bar == 1)
            over_table = (n_bar > 1)
            one_table = one_table * 1
            over_table = over_table * (2*(prob_table > 0.5) - 1)
            n_bar_tilde = n_bar + one_table + over_table
            
            # compute acceptance probability
            #term1 = factorial(n_bar) / factorial(n_bar_tilde)
            term1 = np.zeros((N_alpha, N_alpha), dtype=float)
            term1[n_bar==0] = 1
            term1[n_bar==1] = 1/2
            condition1 = (n_bar > 1) & (prob_table > 0.5)
            condition2 = (n_bar > 1) & (prob_table < 0.5)
            term1[condition1] = 1 / n_bar_tilde[condition1]
            term1[condition2] = n_bar[condition2]

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