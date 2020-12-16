import math
import numpy as np
from scipy.stats import uniform, norm, expon

from .zolotarev import A, B

def ets_sampling_Caron(alpha, sigma, tau, n_sample):
    """it samples W_alpha^* defined at (Caron, 2015, 5.3)

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

    Document
    ----------
    ./doc/ets_sampling.pdf
    """
    assert alpha > 0
    assert sigma > 0 and sigma < 1
    assert tau > 0
    assert type(n_sample) == int

    M = alpha / sigma
    # log_T = log_ets_sampling(alpha=sigma, lambd=M**(1/sigma)*tau, n_sample=n_sample)
    # NOTE : for numerical stability, change lambd ==> lambd_alpha
    log_T = log_ets_sampling(alpha=sigma, lambd_alpha=M*tau**sigma, n_sample=n_sample)
    # T = T * M**(1/sigma)
    T = np.exp(log_T + np.log(M)/sigma)
    
    return T


def log_ets_sampling(alpha, lambd_alpha, n_sample):
    """It samples `lambd` exponential tilted stable(alpha) for `n` times.
    After samples, it logarithms the results for stability of computation.

    Parameters
    ----------
    alpha : float in (0, 1)
        It is related to stable distribution's parameter.

    lambd_alpha : float 
        lambd_alpha = lambd ^ alpha

    n_sample : int
        The number of samples

    Reference
    ----------
    (Devroye, 2009)
    """
    assert alpha < 1 and alpha > 0

    np.random.seed(100)

    #
    # setup
    #
    gamma = (lambd_alpha) * alpha * (1-alpha)
    xi = ((2 + np.sqrt(math.pi/2)) * np.sqrt(2*gamma) + 1) / math.pi
    psi = np.exp( -(gamma * math.pi**2)/8 ) * \
            (2 + np.sqrt(math.pi/2)) * np.sqrt(gamma * math.pi)  / math.pi
    w1 = xi * np.sqrt(math.pi / (2*gamma))
    w2 = 2 * psi * np.sqrt(math.pi)
    w3 = xi * math.pi
    b = (1 - alpha) / alpha

    output = []
    for _ in range(n_sample):
        while True:

            while True:
                #
                # want to sample U ~ g**
                #
                V, W_prime = uniform(0, 1).rvs(), uniform(0, 1).rvs()
                if gamma >= 1:
                    cond = V < w1/(w1+w2)
                    U = abs(norm(0, 1).rvs()) / np.sqrt(gamma) if cond else math.pi * (1 - W_prime**2)
                else:
                    cond = V < w3/(w2+w3)
                    U = math.pi * W_prime if cond else math.pi * (1 - W_prime**2)
                # NOTE: U has density prop to g**

                #
                # From U ~ g**, want to sample U ~ g*
                #
                W = uniform(0, 1).rvs()
                zeta = np.sqrt(B(U, alpha) / B(0, alpha))
                # phi = (np.sqrt(gamma) + alpha*zeta)**(1/alpha)
                # z = phi / (phi - np.sqrt(gamma)**(1/alpha))
                z = 1 / (1 - (1 + alpha*zeta/np.sqrt(gamma))**(-1/alpha) )

                # rho
                num1 = math.pi * np.exp(-lambd_alpha * (1-zeta**(-2)))
                cond = U >= 0 and gamma >= 1
                num2 = np.where(cond, xi * np.exp(-gamma*U**2 / 2), 0.)
                cond = U > 0 and U < math.pi
                num3 = np.where(cond, psi/np.sqrt(math.pi - U), 0.)
                cond = U >= 0 and U <= math.pi and gamma < 1
                num4 = np.where(cond, xi, 0.)
                denominator = (1 + np.sqrt(math.pi/2)) * np.sqrt(gamma) / zeta + z
                rho = num1 * (num2 + num3 + num4) / denominator

                if U < math.pi and W * rho <= 1:
                    Z = W * rho
                    break
            # NOTE: U has density prop to g* and Z is uniformly distributed on [0, 1]

            #
            # From U ~ g*, want to sample g(x, U)
            #
            a = A(U, alpha)
            m = (b / a) ** alpha * lambd_alpha
            delta = np.sqrt(m * alpha / a)
            a1, a2, a3 = delta * np.sqrt(math.pi / 2), delta, z/a
            s = a1 + a2 + a3
            V_prime = uniform(0, 1).rvs()
            N_prime = 0.
            E_prime = 0.
            if V_prime < a1/s:
                N_prime = norm(0, 1).rvs()
                X = m - delta * abs(N_prime)
            elif V_prime < a2/s:
                X = uniform(m, m+delta).rvs()
            else:
                E_prime = expon().rvs()
                X = m + delta + E_prime * a3
            # NOTE: X has density prop to g(x, U)

            #
            # From X ~ g(x, U), want to sample h(x, U)
            #
            E = expon().rvs() # E follows Exponential(1)
            cond1 = X >= 0
            # cond2
            sum_list = [
                a * (X - m),
                # lambd * (X**(-b) - m**(-b)),
                np.exp(1/alpha*np.log(lambd_alpha) - b*np.log(m)) * ((m/X)**b - 1),
                np.where(X < m, -N_prime**2 /2, 0),
                np.where(X > m+delta, -E_prime, 0)
            ]
            cond2 = sum(sum_list) <= E
            if cond1 and cond2:
                output.append( -b*np.log(X) )
                break
    return np.array(output)

if __name__=='__main__':
    ets_sampling_Caron(alpha=300, sigma=0.5, tau=1, n_sample=1)

    # test : ets_sampling_caron
    t = 0.005
    alpha = 300
    sigma = 0.5
    tau = 1
    M = alpha / sigma
    # estimate of Laplace transform of ETS_Caron
    samples = ets_sampling_Caron(alpha=alpha, sigma=sigma, tau=tau, n_sample=5000)
    estimates = np.exp(-t*samples).mean()
    # true(analytic) value of Laplace transform of ETS_Caron
    exponent  = M * (tau**sigma) - (t*M**(1/sigma) + M**(1/sigma)*tau)**sigma
    true_value = np.exp(exponent)
    print(f'true value : {true_value} \n estimates : {estimates}')


    # test : ets_sampling
    lambd = 1
    alpha = 300
    t = 0.05
    log_ets_samples = log_ets_sampling(alpha=alpha, lambd=lambd, n_sample=1000)
    ets_samples = np.exp(log_ets_samples)
    estimates2 = np.exp(-t * ets_samples).mean()
    true_value2 = np.exp(lambd**alpha - (t+lambd)**alpha)
    print(f'true value : {estimates2} \n estimates : {true_value2}')