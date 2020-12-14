import numpy as np
from scipy.stats import expon
from scipy.sparse import find

from simulation import GGPgraphrnd
from mcmc import HMC, MH
from update_n import update_n

Z, _, _ = GGPgraphrnd(alpha=300, sigma=0.5, tau=1)
N_alpha = Z.shape[0]
edge_number = (Z.sum()- sum(Z.diagonal())) / 2 # except self-loop
print(f"Num Nodes : [{N_alpha}] \nNum Edges : [{edge_number}]")

#
# HMC Setting
#
L = 10
efficient_way = True
EPSILON = 1e-4

#
# Step 0: Initialization; phi = (alpha, sigma, tau)
#
np.random.seed(100)
state = {
    'w': expon().rvs(N_alpha),
    'w_star': expon().rvs(N_alpha).mean() * 100,
    'alpha': 1.,
    'sigma': 0.1,
    'tau': 0.1,
    'n': Z.astype(int),
}
state['m'] = np.array(state['n'].astype(int).sum(axis=0))[0] + \
            np.array(state['n'].astype(int).sum(axis=1))[0]
proposal_r_idx, proposal_c_idx, _ = find(Z) # For Step 3, pre-define indices

#
# Step 1: Update w_{1:N_\alpha}
#
state = HMC(state, step_size=EPSILON, num_step=L)

#
# Step 2: Update w_star, phi
#
state = MH(state, sigma_tau=0.1)

#
# Step 3: Update n
#
if efficient_way:
    state['n'] = update_n(Z, state['w'], n_bar=state['n'],
                proposal_idx=(proposal_r_idx, proposal_c_idx))
else:

    state['n'] = update_n(Z, state['w'])