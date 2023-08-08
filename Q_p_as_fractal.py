# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 15:47:47 2021

@author: LENOVO
"""

import matplotlib.pyplot as plt
import Q_p
from fractions import Fraction
import cmath
import math


def projection(x, i, p):
    """
    Calculate the reduction module p^{i} of the
    p-adic number x

    Parameters
    ----------
    x : TYPE:fractions.Fraction or int
        DESCRIPTION: Represent a approximation of p-adic number.
    i : TYPE: int
        DESCRIPTION: represent the power of p^{i} in the reduction.
    p : TYPE: int
        DESCRIPTION: represent a prime number.
    Returns: ans TYPE: fractions.Fraction.
                 DESCRIPTION: Returns the reduction module p^{i} of x.
    -------
    None.

    """
    if x == 0:
        return 0
    if type(x) == int:
        Nume_x = x
        Deno_x = 1
    else:
        Nume_x = x.numerator
        Deno_x = x.denominator

    ans = Fraction(int((Nume_x % ((p**i)*Deno_x))), Deno_x)
    return ans


def rho(x, n, m, p):
    """


    Parameters
    ----------
    x : TYPE:  fractions.Fraction or int
        DESCRIPTION: Represent a p-adic number.
    n : TYPE: int
        DESCRIPTION.
    m : TYPE: int
        DESCRIPTION.
    p : TYPE: int
        DESCRIPTION. Represent a prime.

    Returns: ans: TYPE: fractions.Fraction.
                  DESCRIPTION: represent the partial sum of its p-adic coefictions
                  from n-m to n (n to n-m) when m>=0 (m<0).
    -------
    None.

    """
    if n >= 0:
        if m >= n:
            return projection(x, n+1, p)
        else:
            return projection(x, n+1, p)-projection(x, n-m, p)

    if m < 0:
        return projection(x, n-m+1, p)-projection(x, n, p)
    elif m >= 0:
        return projection(x-projection(x, n-m, p), n+1, p)


def chi(x, n, m, p):
    """


    Parameters
    ----------
    x : TYPE:fractions.Fraction or int
        DESCRIPTION. Represent a p-adic number.
    m : TYPE: int
        DESCRIPTION.
    n : TYPE: int
        DESCRIPTION.
    p : TYPE: int
        DESCRIPTION.: Represent a prime

    Returns: ans:   TYPE: complex
                    DESCRIPTION: returns  the complex exp(2*pi/p**(n+1)*approximation of x)
    -------
    None.

    """
    i = 1j
    pi = cmath.pi
    constant = (2*pi*i)/(p**(n+1))
    return cmath.exp(constant*rho(x, n, m, p))


def gamma(x, m, s, p, K):
    """


    Parameters
    ----------
    x : TYPE: fraction.Fraction or int
        DESCRIPTION: Represents a p-adic number.
    m : TYPE: int
        DESCRIPTION: Represent an non-zero itenger as a parameter.
    s : TYPE: complex
        DESCRIPTION: Represents an complex number with module less than 1.
    p : TYPE: int
        DESCRIPTION: Represents an prime number.
    K : TYPE: int
        DESCRIPTION. Represent the group G_K.

    Raises
    ------
    NameError
        DESCRIPTION: When the complex number s has module lager or equal to 1.

    Returns: ans
    -------
    TYPE complex
        DESCRIPTION: Represents the complex number asociated to x.

    """
    if x == 0:
        return 1/(1-s)

    if type(x) == int:
        valuation_x = Q_p.valuation(x, p)
    else:
        valuation_x = Q_p.valuation(
            x.numerator, p)-Q_p.valuation(x.denominator, p)
    ans = 0

    if math.sqrt(s.real**2+s.imag**2) >= 1:

        raise NameError("the complex number s must have module less than 1")

    for n in range(valuation_x, m+K):
        ans += (s**n)*chi(x, n, m, p)

    if valuation_x == 1:
        return ans+1
    else:
        return ans+((1-s**(valuation_x))/(1-s))+(s**(m+K+1)/(1-s))


""" ----------------- Plot of p-adic numbers -------------"""


def Christiakov_emmending(Z_k, m, s):
    """


    Parameters
    ----------
    Z_k : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    s : TYPE
        DESCRIPTION.

    Returns pair with the x and y sitions of e
            each p-adic number
    -------
    x_list : TYPE list
        DESCRIPTION.
    y_list : TYPE list
        DESCRIPTION.

    """
    x_list = []
    y_list = []

    for x in Z_k:
        p = Z_k.get_prime()
        K = Z_k.get_radio()
        complex_x = gamma(x, m, s, p, K)
        Real_x = complex_x.real
        Imgen_x = complex_x.imag
        x_list.append(Real_x)
        y_list.append(Imgen_x)
    return (x_list, y_list)


"""
p = 3
m = 1
s = 0.4999
K = 4
Z_k = Q_p.Z_N_group(p, K)

x_list, y_list = Christiakov_emmending(Z_k, m, s)
plt.scatter(x_list, y_list, s=0.5, color="green")
"""
