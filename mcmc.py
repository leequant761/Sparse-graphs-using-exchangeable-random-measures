import numpy as np
from numpy import log
from numpy.random import lognormal, gamma
from scipy.stats import norm
from scipy.special import gamma as GAMMA

from ets import ets_sampling_Caron

def grad_log_posterior(state):
    """
    Reference
    ----------
    (Caron, 2015) (49)
    """
    # NOTE: I think the below expression is correct
    # grad = m - (1+state['sigma']) - state['w']*(state['tau'] + 2*sum(state['w'] + 2*state['w_star']))
    grad = state['m'] - state['sigma'] - \
        state['w'] * (state['tau']+2*sum(state['w']+2*state['w_star']))
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
        'w_star': state['w_star'],
        'm': state['m'],
        'alpha': state['alpha'],
        'sigma': state['sigma'],
        'tau': state['tau']
    }
    init_p = norm().rvs(N_alpha)
    h_state['p'] = init_p +  step_size/2 * grad_log_posterior(state)
    for l in range(1, num_step):
        h_state['w'] = h_state['w'] * np.exp(step_size * h_state['p'])
        h_state['p'] = h_state['p'] + step_size*grad_log_posterior(h_state)
    h_state['w'] = h_state['w'] * np.exp(step_size * h_state['p'])
    h_state['p'] = -(h_state['p'] + step_size/2 * grad_log_posterior(h_state))

    # Accept the proposal with probability with min(1,r)
    term1 = (h_state['m'] - h_state['sigma']) * (np.log(h_state['w']) - np.log(state['w']))
    term1 = term1.sum()
    term2 = -(h_state['w'].sum() + state['w_star'])**2
    term3 = (state['w'].sum() + state['w_star'])**2
    term4 = -h_state['tau'] * (state['w'].sum() + state['w'].sum())
    term5 = -1/2 * (h_state['p']**2-init_p**2).sum()
    r = np.exp(term1 + term2 + term3 + term4 + term5)
    is_accept = np.random.binomial(1, min(1, r))

    if is_accept:
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
    freq_term1 = m_state['tau'] + 2*sum(state['w']) + state['w_star']
    m_state['alpha'] = gamma(
        N_alpha, 
        (freq_term1**m_state['sigma'] - state['tau']**m_state['sigma']) / m_state['sigma']
        )
    m_state['w_star'] = ets_sampling_Caron(m_state['alpha'],
                                            m_state['sigma'], freq_term1, 1)[0]
    freq_term2 = state['tau'] + 2*sum(state['w']) + m_state['w_star']

    # compute acceptance probability
    term1 = (sum(state['w']) + state['w_star'])**2 - (sum(state['w']) + state['w_star'])**2
    term2 = -(m_state['tau']-state['tau']+2*state['w_star']-2*m_state['w_star']) * state['w'].sum()
    term3 = (-m_state['sigma']+state['sigma']) * log(state['w']).sum()
    term4_n1 = log(GAMMA(1-state['sigma'])) - log(state['sigma'])
    term4_n2 = log(freq_term2**state['sigma'] - state['tau']**state['sigma'])
    term4_n = term4_n1 + term4_n2
    term4_d1 = log(GAMMA(1-m_state['sigma'])) - log(m_state['sigma'])
    term4_d2 = log(freq_term1**m_state['sigma'] - m_state['tau']**m_state['sigma'])
    term4_d = term4_d1 + term4_d2
    term4 = N_alpha * (term4_n - term4_d)
    r = np.exp(sum([term1, term2, term3, term4]))
    acceptance_pbt = min(1, r)
    is_accept = np.random.binomial(1, acceptance_pbt)
    
    if is_accept:
        state['alpha'] = m_state['alpha']
        state['sigma'] = m_state['sigma']
        state['tau'] = m_state['tau']
        state['w_star'] = m_state['w_star']

    return state