"""
"""
import numpy as np



"""
Question 1
"""
"a"

def S_AR(f, phis, sigma2):
    
    # set value of p from phis
    p = len(phis)
    
    # # generate the vector (1, -e^(-i*2*pi*f), ..., -e^(-i*2*pi*f*p))
    # v1 = np.ones(p+1, dtype = 'complex') * np.exp(--1j * 2 * np.pi)
    
    # create input vector required to construct the vandermonde matrix
    vec = np.exp(--1j * 2 * np.pi) ** f
    
    # create the Vandermonde matrix  V and the vector x
    V = np.vander(vec, p+1)
    
    x = np.ones(p+1)
    x[1:] = - phis
    
    # evaluate S using the parametric form of the sdf
    S = sigma2 / np.abs(V @ x)
    
    return S