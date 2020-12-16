import random

import numpy as np
from numpy import log
from numpy.random import lognormal, gamma
from scipy.stats import norm
from scipy.special import gamma as GAMMA

from ets import ets_sampling_Caron
from rnd import finite_crm_Caron

def grad_log_posterior(state):
    """
    Reference
    ----------
    (Caron, 2015) (49)
    """
    # NOTE: I think the below expression is correct
    # grad = m - (1+state['sigma']) - state['w']*(state['tau'] + 2*sum(state['w'] + 2*state['w_star']))
    grad = state['m'] - state['sigma'] - \
        state['w'] * (state['tau'] + 2*state['w'].sum() + 2*state['w_star'])
    return grad

def HMC(state, step_size, num_step):
    """
    Reference
    ----------
    (Caron, 2015) E.1. Step 1
    """
    N_alpha = len(state['w'])
    h_state = {
        'w': state['w'],
        'log_w': np.log(state['w']),
        'w_star': state['w_star'],
        'm': state['m'],
        'sigma': state['sigma'],
        'tau': state['tau']
    }

    init_p = norm().rvs(N_alpha)
    h_state['p'] = init_p +  step_size/2 * grad_log_posterior(state)
    for l in range(1, num_step):
        # h_state['w'] = h_state['w'] * np.exp(step_size * h_state['p'])
        h_state['log_w'] = h_state['log_w'] + step_size * h_state['p']
        h_state['p'] = h_state['p'] + step_size*grad_log_posterior(h_state)
    h_state['log_w'] = h_state['w'] + step_size * h_state['p']
    h_state['p'] = -(h_state['p'] + step_size/2 * grad_log_posterior(h_state))
    h_state['w'] = np.exp(h_state['log_w'])

    # Accept the proposal with probability with min(1,r)
    term1 = (state['m'] - state['sigma']) * (h_state['log_w'] - np.log(state['w']))
    term1 = term1.sum()
    term2 = -(h_state['w'].sum() + state['w_star'])**2
    term3 = (state['w'].sum() + state['w_star'])**2
    term4 = -state['tau'] * (h_state['w'].sum() - state['w'].sum()) # paper errata
    term5 = -1/2 * (h_state['p']**2-init_p**2).sum()
    log_r = term1 + term2 + term3 + term4 + term5

    if np.isnan(log_r):
        log_r = -np.inf

    if np.log(np.random.uniform()) < log_r:
        state['w'] = h_state['w']

    return state

def MH(state, sigma_tau):
    """
    Reference
    ----------
    (Caron, 2015) E.1. Step 2
    """
    N_alpha = len(state['w'])
    m_state = {
        'tau': lognormal(log(state['tau']), sigma_tau),
        'sigma': 1-lognormal(log(1-state['sigma']), sigma_tau),
    }
    freq_term1 = m_state['tau'] + 2*state['w'].sum() + state['w_star']
    m_state['alpha'] = gamma(
        N_alpha, 
        m_state['sigma'] / (freq_term1**m_state['sigma'] - state['tau']**m_state['sigma'])
        )
    if m_state['sigma'] > 0:
        m_state['w_star'] = ets_sampling_Caron(m_state['alpha'], 
                                            m_state['sigma'], freq_term1, 1)[0]
    else:
        m_state['w_star'] = finite_crm_Caron(m_state['alpha'], m_state['sigma'], 
                                m_state['tau']+2*state['w'].sum()+state['w_star'])
    
    # freq_term2 = state['tau'] + 2*sum(state['w']) + m_state['w_star']

    # compute acceptance probability
    term1 = (sum(state['w']) + state['w_star'])**2 - (sum(state['w']) + m_state['w_star'])**2
    term2 = -(m_state['tau']-state['tau']-2*m_state['w_star']+2*state['w_star']) * state['w'].sum()
    term3 = (-m_state['sigma']+state['sigma']) * log(state['w']).sum()
    
    # term4's numerator
    term4_n1 = log(GAMMA(1-state['sigma']))
    # term4_n2 = log((freq_term2**state['sigma'] - state['tau']**state['sigma']) / state['sigma'])
    stable_term2 = 1 + (2*state['w'].sum() + m_state['w_star']) / state['tau']
    stable_term2 = (stable_term2 ** state['sigma'] - 1) / state['sigma']
    term4_n2 = log(stable_term2) + state['sigma'] * log(state['tau'])
    term4_n = term4_n1 + term4_n2
    
    # term4's denominator
    term4_d1 = log(GAMMA(1-m_state['sigma']))
    # term4_d2 = log(freq_term1**m_state['sigma'] - m_state['tau']**m_state['sigma'])
    stable_term1 = 1 + (2*state['w'].sum() + state['w_star']) / m_state['tau']
    stable_term1 = (stable_term1 ** m_state['sigma'] - 1) / m_state['sigma']
    term4_d2 = log(stable_term1) + m_state['sigma'] * log(m_state['tau'])
    term4_d = term4_d1 + term4_d2

    term4 = N_alpha * (term4_n - term4_d)
    
    log_r = term1 + term2 + term3 + term4
    if log(random.random()) < log_r:
        state['alpha'] = m_state['alpha']
        state['sigma'] = m_state['sigma']
        state['tau'] = m_state['tau']
        state['w_star'] = m_state['w_star']

    return state