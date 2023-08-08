"""
@author: Brian Zambrano
Classes:

    G_N_group
    Z_N_group

Functions:
    valuation(int,int)->int
    norm_p(float,int) -> float
"""
from fractions import Fraction
import numpy as np
import math


def valuation(n, p):
    """
    Returns the maximum power of p that divides n.

    Parameters
    ----------
    n : TYPE: int
        DESCRIPTION: A intenger number.
    p : TYPE: int
        DESCRIPTION: A prime number.

    Returns
    -------
    TYPE: int
    DESCRIPTION: ans: the maximum power of p that divides n.

    """

    n = abs(n)

    if n % p != 0:
        return 0
    else:
        ans = 0
        while n % p == 0:
            n = n/p
            ans += 1
        return ans


def norm_p(x, p):
    """
    Returns the p-adic norm of x.

    Parameters
    ----------
    x : TYPE: float/int
        DESCRIPTION: A rational number.
    p : TYPE: int
        DESCRIPTION: A prime number.

    Returns
    -------
    TYPE: float
    DESCRIPTION: p**(-valuation_x): the p-adic norm of x.

    """

    if x == 0:
        return 0
    elif type(x) == int:
        return p**(-valuation(x, p))
    else:
        x = Fraction(x)
        n = x.numerator
        d = x.denominator
        valuation_x = valuation(n, p)-valuation(d, p)
        return p**(-valuation_x)


def G_N_group_aux(p, N):
    """
    Returns the p-adic group G_K elments.

    Parameters
    ----------
    p : TYPE: int
        DESCRIPTION: A prime number.
    N : TYPE: int
        DESCRIPTION: A natural number,
        represents the parameter of the group G_N.

    Returns
    -------
    TYPE: numpy array
    DESCRIPTION: ans1: the p-adic group G_K elments.

    """
    def Fraction_lim(a, p, N):
        return Fraction(a).limit_denominator(p**N)

    pn_Fraction_lim = np.vectorize(Fraction_lim)

    ans1 = np.array([])
    l = 1
    for j in range(0, p):
        ans1 = np.append(ans1, j*(p**(2*N-l)))

    l = 2
    while l <= 2*N:

        ans2 = ans1.copy()
        for j in range(1, p):
            ans1 = np.concatenate([ans1, ans2+j*(p**(2*N-l))])
        l += 1
    ans1 = pn_Fraction_lim(ans1, p, 1)
    ans1 = ans1*Fraction(1, p**N)
    return ans1


class G_N_group(object):
    """Define the object p^{-N}Z_p/p^N Z_p"""

    def __init__(self, p, N):
        self.p = p
        self.N = N
        G_N = G_N_group_aux(self.p, self.N)
        self.G_N = G_N
        self.index = -1

    def get_prime(self):
        return self.p

    def get_radio(self):
        return self.N

    def __len__(self):
        p = self.p
        N = self.N
        return p**(2*N)

    def __str__(self):
        G_N = self.G_N
        return str(G_N)

    def __iter__(self):
        for a in self.G_N:
            yield a


def Z_N_group_aux(p, N):
    """
    Returns the p-adic group Z_N elments.

    Parameters
    ----------
    p : TYPE: int
        DESCRIPTION: A prime number.
    N : TYPE: int
        DESCRIPTION: A natural number,
        represents the parameter of the group Z_N.

    Returns
    -------
    TYPE: numpy array
    DESCRIPTION: ans1: the p-adic group G_K elments.

    """

    ans1 = np.array([], dtype=int)
    l = 1
    for j in range(0, p):
        ans1 = np.append(ans1, j*(p**(N-l)))

    l = 2
    while l <= N:

        ans2 = ans1.copy()

        for j in range(1, p):
            ans1 = np.concatenate([ans1, ans2+j*(p**(N-l))])
        l += 1

    return ans1


class Z_N_group(object):
    """Define the object Z_p/p^NZ_p"""

    def __init__(self, p, N):
        self.p = p
        self.N = N
        Z_N = Z_N_group_aux(self.p, self.N)
        self.Z_N = Z_N

    def get_prime(self):
        return self.p

    def get_radio(self):
        return self.N

    def __len__(self):
        p = self.p
        N = self.N
        return p**(N)

    def __str__(self):
        Z_N = self.Z_N
        return str(Z_N)

    def __iter__(self):
        for a in self.Z_N:
            yield a


def Monna_map(x, p):
    """
    We apply the Monna map to p-adic intengers

    Parameters
    ----------
    x : TYPE  int
        DESCRIPTION:
            x is an intanger in Z_p/p^KZ_p

    Returns float. ratinal number less than 1
    -------
    None.

    """
    if x == 0:
        return 0
    else:
        k = int(math.log(x, p))
        m_x = int(x/p**(k))  # x_{k}
        x_i_p = m_x * p**(k)
        for i in range(1, k+1):
            x_i_1 = int((x-x_i_p)/p**(k-i))
            m_x += x_i_1*p**(i)
            x_i_p += x_i_1*p**(k-i)  # x_{k-1}*p**(k-1)+x_{k}*p**(k)
        return m_x*(p**(-k-1))
