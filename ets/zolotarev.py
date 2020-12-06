import numpy as np
from numpy import sin

def A(u, alpha):
    """Used exactly same notation with (Devroye, 2009)
    """
    numerator = (sin(alpha * u)**alpha) * (sin((1-alpha)*u)**(1-alpha))
    denominator = sin(u)
    return (numerator / denominator) ** (1 / (1-alpha))

def B_(u, alpha):
    """Just for test
    """
    return A(u, alpha) ** (-(1-alpha))

def B(u, alpha):
    """Used exactly same notation with (Devroye, 2009)
    """
    if u > 0:
        numerator = (sin(alpha * u)**alpha) * (sin((1-alpha)*u)**(1-alpha))
        denominator = sin(u)
        return denominator / numerator
    elif u == 0:
        return alpha**(-alpha) * (1-alpha)**(-(1-alpha))
    else:
        raise ValueError(f'u={u} should be positive')