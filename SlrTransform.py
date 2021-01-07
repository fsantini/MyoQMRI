#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 09:32:29 2021

Ported from Brian Hargreaves http://mrsrl.stanford.edu/~brian/bloch/
"""

import numpy as np

def slr(pulse):
    """
    Shinnar-LeRoux transform

    Parameters
    ----------
    pulse : np.array
        series of rotations. Make sure that sum(pulse) == fa

    Returns
    -------
    A,B coefficients of the SLR transform

    """
    
    pulse_len = len(pulse);
    
    C = np.cos(np.abs(pulse/2))
    S = 1j * np.exp(1j*np.angle(pulse)) * np.sin(np.abs(pulse/2))
    
    AB = np.array([[1],
                   [0]], dtype=np.complex64)
    

    for k in range(pulse_len):
        AB = np.stack([np.append(C[k]*AB[0,:],0)-np.append(0, np.conj(S[k])*AB[1,:]),
                   np.append(S[k]*AB[0,:],0)+np.append(0, np.conj(C[k])*AB[1,:])])
    
    A = AB[0,:]
    B = AB[1,:]
    
    if np.abs(A[-1]) > .0001:
        print('Warning:  Last term of A not 0')
    
    if np.abs(B[-1]) > .0001:
        print('Warning:  Last term of B not 0')
        
    
    return A[:-1], B[:-1]