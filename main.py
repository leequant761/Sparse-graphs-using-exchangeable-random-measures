import datetime
import os
import argparse

import numpy as np
from scipy.stats import gamma, uniform
from scipy.sparse import find, coo_matrix, triu
import pandas as pd

from mcmc import HMC, MH
from update_n import update_n
from custom_io import load_graph


def main(args):
    np.random.seed(110)
    graph, directed = load_graph(args.dataset)
    if not directed:
        # By convention, we just need to infer upper triangle's n_ij
        graph = triu(graph + graph.T > 0)
    N_alpha = graph.shape[0]
    edge_number = len(graph.data)
    print(f"[{args.dataset}] Num Nodes : {N_alpha} \t Num Edges : {edge_number}")
    
    # other settings
    proposal_r_idx, proposal_c_idx, _ = find(graph) # For Step 3, pre-define indices
    epsilon = args.epsilon / (N_alpha ** 0.25)
    history = {
        'w_star': [],
        'alpha': [],
        'sigma': [],
        'tau': []
    }
    now = datetime.datetime.now().strftime("%m%d-%H%M")

    #
    # Step 0: Initialization; phi = (alpha, sigma, tau)
    #
    rates = np.array([])
    np.random.seed(100)
    state = {
        'w': gamma(a=1, scale=1/1).rvs(N_alpha),
        'w_star': gamma(a=1, scale=1/1).rvs(),
        'alpha': 100*uniform.rvs(),
        'sigma': 2*uniform.rvs()-1,
        'tau': 10*uniform.rvs(),
        'n': graph.astype(int),
    }
    if not directed:
        state['n'].data = np.random.randint(1, 10+1, len(state['n'].data))
    state['m'] = np.array(state['n'].sum(axis=0))[0] + \
                np.array(state['n'].sum(axis=1))[:,0]

    for epoch in range(args.n_iter):
        #
        # Step 1: Update w_{1:N_\alpha}
        #
        state, a_rate = HMC(state, step_size=epsilon, num_step=args.L)
        if epoch < args.n_adapt:
            rates = np.append(rates, a_rate)
            epsilon = np.exp(np.log(epsilon) + args.adapt_step_size*(rates.mean() - args.adapt_pbt))

        #
        # Step 2: Update w_star, phi
        #
        state = MH(state, sigma_tau=0.02)

        #
        # Step 3: Update n
        #
        if not directed:
            state['n'].data = update_n(state['w'], n_bar=np.array(state['n'].data), 
                                        proposal_idx=(proposal_r_idx, proposal_c_idx))
            state['m'] = np.array(state['n'].sum(axis=0))[0] + \
                        np.array(state['n'].sum(axis=1))[:,0]

        #
        # Notification & Save
        #
        if epoch % 100 == 0:
            print(f'EPOCH : {epoch}')
            print('w_star: {0:.2f} \n alpha: {1:.2f} \n sigma: {2:.2f} \n tau: {3:.2f}'\
            .format(state['w_star'], state['alpha'], state['sigma'], state['tau']))
        if (epoch > args.n_burn) and  (epoch % args.thin) == 0:
            history['w_star'].append(state['w_star'])
            history['alpha'].append(state['alpha'])
            history['sigma'].append(state['sigma'])
            history['tau'].append(state['tau'])

    if not 'results' in os.listdir('.'):
        os.mkdir('./results')
    os.mkdir(f'./results/{now}')
    pd.DataFrame(history).to_csv(f'./results/{now}/results.csv')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='parse args')
    # dataset
    parser.add_argument('--dataset', default='simul', type=str, 
                        help='which dataset to use (ex: simul, email, airport, powergrid)')
    # MCMC settings
    parser.add_argument('--n_iter', default=50000, type=int,
                        help='MCMC parameter: the number of MCMC iterations')
    parser.add_argument('--n_burn', default=10000, type=int,
                        help='MCMC parameter: the number of burn-in iterations')
    parser.add_argument('--thin', default=2, type=int,
                        help='MCMC parameter: thinning of the MCMC output')
    # HMC settings
    parser.add_argument('--L', default=10, type=int,
                        help='HMC parameter: the number of leapfrog steps')
    parser.add_argument('--epsilon', default=0.1, type=float,
                        help='HMC parameter: leapfrog stepsize')
    parser.add_argument('--n_adapt', default=12500, type=int,
                        help='HMC parameter: the number of adapation of leapfrog stepsize')
    parser.add_argument('--adapt_pbt', default=0.6, type=float,
                        help='HMC parameter: adapation so as to obtain this acceptance rate')
    parser.add_argument('--adapt_step_size', default=0.01, type=float,
                        help='HMC parameter: adaptation stepsize')

    args = parser.parse_args()
    main(args)
