import datetime
import os

import numpy as np
from scipy.stats import gamma, uniform
from scipy.sparse import find
import pandas as pd

from simulation import GGPgraphrnd
from mcmc import HMC, MH
from update_n import update_n

Z, _, _ = GGPgraphrnd(alpha=300, sigma=0.5, tau=1)
N_alpha = Z.shape[0]
edge_number = (Z.sum()- sum(Z.diagonal())) / 2 # except self-loop
print(f"Num Nodes : [{N_alpha}] \nNum Edges : [{edge_number}]")

now = datetime.datetime.now().strftime("%m%d-%H%M")

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
    'w': gamma(a=1, scale=1/1).rvs(N_alpha),
    'w_star': gamma(a=1, scale=1/1).rvs(),
    'alpha': 100*uniform.rvs(),
    'sigma': 0.1,
    'tau': 10*uniform.rvs(),
    'n': Z.astype(int),
}
state['n'].data = np.random.randint(1, 10+1, len(state['n'].data))
state['m'] = np.array(state['n'].astype(int).sum(axis=0))[0] + \
            np.array(state['n'].astype(int).sum(axis=1))[0]
proposal_r_idx, proposal_c_idx, _ = find(Z) # For Step 3, pre-define indices

history = {
    'w_star': [],
    'alpha': [],
    'sigma': [],
    'tau': []
}
for epoch in range(10000):
    #
    # Step 1: Update w_{1:N_\alpha}
    #
    state = HMC(state, step_size=EPSILON, num_step=L)

    #
    # Step 2: Update w_star, phi
    #
    state = MH(state, sigma_tau=0.02)

    #
    # Step 3: Update n
    #
    if efficient_way:
        update_n(Z, state['w'], n_bar=state['n'].data, 
                proposal_idx=(proposal_r_idx, proposal_c_idx))
    else:
        state['n'] = update_n(Z, state['w'])

    #
    # Notification & Save
    #
    if epoch % 100 == 0:
        print(f'EPOCH : {epoch}')
        print('w_star: {0:.2f} \n alpha: {1:.2f} \n sigma: {2:.2f} \n tau: {3:.2f}'\
        .format(state['w_star'], state['alpha'], state['sigma'], state['tau']))
    history['w_star'].append(state['w_star'])
    history['alpha'].append(state['alpha'])
    history['sigma'].append(state['sigma'])
    history['tau'].append(state['tau'])

if not 'results' in os.listdir('.'):
    os.mkdir('./results')
pd.DataFrame(history).to_csv(f'./{now}results.csv')