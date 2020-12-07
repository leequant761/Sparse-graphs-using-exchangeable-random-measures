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
    T = ets_sampling(alpha=sigma, lambd=M**(1/sigma)*tau, n_sample=n_sample)
    T = T * M**(1/sigma)
    
    return T


def ets_sampling(alpha, lambd, n_sample):
    """It samples `lambd` exponential tilted stable(alpha) for `n` times.

    Parameters
    ----------
    alpha : float in (0, 1)
        It is related to stable distribution's parameter.

    lambd : float 
        Number of (training) examples per class in each task. This corresponds
        to `k` in `k-shot` classification.

    n_sample : int
        The number of samples

    Reference
    ----------
    (Devroye, 2009)
    """

    np.random.seed(100)

    #
    # setup
    #
    gamma = (lambd ** alpha) * alpha * (1-alpha)
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
                phi = (np.sqrt(gamma) + alpha*zeta)**(1/alpha)
                z = phi / (phi - np.sqrt(gamma)**(1/alpha))
                # rho
                num1 = math.pi * np.exp(-lambd**alpha * (1-zeta**(-2)))
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
            m = (b * lambd / a) ** alpha
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
                E_prime = expon(1).rvs()
                X = m + delta + E_prime * a3
            # NOTE: X has density prop to g(x, U)

            #
            # From X ~ g(x, U), want to sample h(x, U)
            #
            E = -np.log(Z) # E follows Exponential(1)
            cond1 = X >= 0
            # cond2
            sum_list = [
                a * (X - m),
                lambd * (X**(-b) - m**(-b)),
                np.where(X < m, -N_prime**2 /2, 0),
                np.where(X > m+delta, -E_prime, 0)
            ]
            cond2 = sum(sum_list) <= E
            if cond1 and cond2:
                output.append( 1/(X**b) )
                break
    return np.array(output)

if __name__=='__main__':
    ets_sampling_Caron(alpha=300, sigma=0.5, tau=1, n_sample=1)