import numpy as np


def vectorize_function(f, Z_k):
    """
    Parameters
    ----------
    f : TYPE callable object or np.array
        DESCRIPTION.
    Z_k : TYPE
        DESCRIPTION.

    Returns 1d numpy array
    -------
    None.

    """
    if type(f) == np.ndarray:
        return f
    elif type(f) == int or type(f) == float:
        return f*np.ones(len(Z_k))
    else:
        vector = []
        for a in Z_k:
            vector.append(f(a))
        return np.array(vector)


def p_adic_Convolution_matrix(g, Z_k):
    p = Z_k.get_prime()
    k = Z_k.get_radio()
    if type(g) == int or type(g) == float:
        return (p**(-k))*g*np.ones((p**k, p**k))
    else:
        vect_g = []
        index = np.array([i for i in Z_k], dtype=int)
        Dic_index = dict()
        for a in index:
            Dic_index[a] = g(a)
        vect_g = np.array(vect_g)
        matrix_g = np.zeros((p**k, p**k))
        num_row = 0
        for j in index:
            New_Values = (index - j) % p**k
            matrix_g[num_row] = np.array([Dic_index[a] for a in New_Values])
            num_row += 1
        return p**(-k)*matrix_g


def vect2p_adic_conv(V, Z_k):
    p = Z_k.get_prime()
    k = Z_k.get_radio()
    vect_g = []
    index = np.array([i for i in Z_k], dtype=int)
    Dic_index = dict()
    for a in range(len(Z_k)):
        Dic_index[index[a]] = V[a]
    vect_g = np.array(vect_g)
    matrix_g = np.zeros((p**k, p**k))
    num_row = 0
    for j in index:
        New_Values = (index - j) % p**k
        matrix_g[num_row] = np.array([Dic_index[a] for a in New_Values])
        num_row += 1
    return matrix_g
