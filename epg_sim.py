#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    This file is part of MyoQMRI.

    MyoQMRI is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Foobar is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
    
    Copyright 2019 Francesco Santini <francesco.santini@unibas.ch>

Based on:
    Weigel, M. (2015), Extended phase graphs: Dephasing, RF pulses, and echoes ‐ pure and simple. J. Magn. Reson. Imaging, 41: 266-295. doi:10.1002/jmri.24619    
"""

from __future__ import print_function

import timeit
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import numpy.linalg as linalg
from scipy.optimize import minimize

np.set_printoptions(precision = 4)

TMatrices = {}

def cpmg(N_in, excitation_alpha_in, refocus_alpha_in, EchoSpacing, T1, T2, magPreparePulse = False):
    #alpha_in = 120 # refocusing pulse FA
    #N_in = 10 # echo train length
    #T1 = 1000.0
    #T2 = 50.0
    #EchoSpacing = 20.0
    
    excitation_alpha_in = float(excitation_alpha_in)
    alpha_in = float(refocus_alpha_in)
    EchoSpacing = float(EchoSpacing)
    T1 = float(T1)
    T2 = float(T2)
    
    
    alpha_in = np.radians(alpha_in)
    fa = np.ones((N_in), dtype = np.complex)*alpha_in # flip angle array
    
    N     = len(fa)
    Nt2   = 2*N                                 
    Nt2p1 = Nt2+1
    
    if N>1 and magPreparePulse:
        fa[0] = (np.pi+fa[1])/2 # first refocusing pulse
        
    if (T1 == 0):
        E1 = 1.0
    else:
        E1 = np.exp(-EchoSpacing/T1/2.0)
        
    if (T2 == 0):
        E2 = 1.0
    else:
        E2 = np.exp(-EchoSpacing/T2/2.0)
        
        
    RelaxMatrix = np.concatenate( [np.ones((2,Nt2p1))*E2, np.ones((1,Nt2p1))*E1], axis = 0)
        
    # Generate state matrices Omega before and after RF: Eq.[26] in EPG-R
    Omega_preRF  = np.zeros((3,Nt2p1), dtype = np.complex)
    Omega_postRF = np.zeros((3,Nt2p1), dtype = np.complex)
    
    # CPMG condition, with magnetization on +x
    Omega_postRF[0,0] = np.sin(np.radians(excitation_alpha_in)) # 1
    Omega_postRF[1,0] = np.sin(np.radians(excitation_alpha_in)) # 1
    Omega_postRF[2,0] = np.cos(np.radians(excitation_alpha_in))

    
    F0_vector_out = np.zeros( (N), dtype = np.complex )
    
    def dephase(stateMatrix):
        outMatrix = np.zeros_like(stateMatrix)
        # Dephasing - only XY plane
        outMatrix[0,1:] = stateMatrix[0,:-1]
        outMatrix[1,:-1] = stateMatrix[1,1:]
        outMatrix[0,0] = np.conj(outMatrix[1,0])
        outMatrix[2,:] = stateMatrix[2,:]
        #print outMatrix
        return outMatrix
        
    def relax(stateMatrix, relaxMatrix):
        outMatrix = np.multiply(stateMatrix, relaxMatrix)
        outMatrix[2,0] += 1 - relaxMatrix[2,0]
        return outMatrix
    
    for pn in range(N):
        T = None
        # only recalculate matrix if the FA changed
        if fa[pn] not in TMatrices:
            T = np.zeros((3,3), dtype = np.complex)
            T[0,0] =       np.cos(fa[pn]/2.0)**2
            T[0,1] =       np.sin(fa[pn]/2.0)**2
            T[0,2] = -1j * np.sin(fa[pn])
            
            T[1,0] =       np.sin(fa[pn]/2.0)**2
            T[1,1] =       np.cos(fa[pn]/2.0)**2
            T[1,2] = +1j * np.sin(fa[pn])
            
            T[2,0] = -.5j* np.sin(fa[pn])
            T[2,1] = +.5j* np.sin(fa[pn])
            T[2,2] =       np.cos(fa[pn])
            TMatrices[fa[pn]] = T
        else:
            T = TMatrices[fa[pn]]
        
        # relaxation/recovery
        Omega_preRF = relax(Omega_postRF, RelaxMatrix)

        Omega_preRF = dephase(Omega_preRF)
        
        # RF
        Omega_postRF = np.dot(T,Omega_preRF)
        
        # relaxation/recovery
        Omega_postRF = relax(Omega_postRF, RelaxMatrix)
        
        
        Omega_postRF = dephase(Omega_postRF)
        
        F0_vector_out[pn] = np.conj(Omega_postRF[1,0])
        #print F0_vector_out

    return F0_vector_out
    

def calcSliceprof(nomFA, tbw): 
    t = np.linspace(-tbw/2, tbw/2, 1000)
    pulse = np.multiply(np.hanning(len(t)), np.sinc(t))
    total = np.sum(pulse)
    pulse = np.radians(pulse / total * nomFA)
    

    # alternative a bit less precise but errors ~1deg
    h = np.abs(fft.fft(pulse,5210*2))
    
    sliceprof = np.abs(h[0:199])
    sliceprof = sliceprof/np.max(sliceprof)*nomFA
    return sliceprof

# binning of the slice profile
def reduceSliceProf(sliceprof, bins):
    lastVal = np.argwhere( sliceprof > 0.5 ).max()
    sliceprof = sliceprof[:lastVal]
    
    binsize = np.ceil(len(sliceprof) / bins);
    sliceprof_out = np.zeros((bins,1));
    
    for i in range(bins):
        startIndex = i*binsize
        endIndex = (i+1)*binsize-1
        if endIndex > len(sliceprof):
            endIndex = len(sliceprof)-1
        sliceprof_out[i] = sliceprof[startIndex:endIndex].mean()
    
    return sliceprof_out
        

if __name__ == '__main__':
    
    
    fatSignal = np.array([1.29E+03,1.62E+03,1.48E+03,1.40E+03,1.31E+03,1.23E+03,1.16E+03,1.10E+03,1.02E+03,979,918,877,827,791,749,717,679])
    waterSignal = np.array([249,257,194,157,125,101,81.4,67.7,55.3,45.5,39,33.5,27.5,26.1,19.6,19.6,16.1])
    
    echoSpacing = 9.5
    NEchoes = len(fatSignal)
    
    T1w = 1400
    T1f = 365

    t2Lim = (10,300)
    b1Lim = (0.5,1.2)

    
    sliceProf90 = reduceSliceProf(calcSliceprof(90, 2.0),10)
    sliceProf90 = np.pad(sliceProf90, ((0,2),(0,0)), 'constant')
    sliceProf180 = reduceSliceProf(calcSliceprof(180, 2.0),12) # maybe take into account larger slice for refocusing
    
    plt.ion()
    
    def _signalCalc(T1, T2, B1Factor):
        signal = np.zeros((NEchoes), dtype=np.complex)
        for curFAIndex in range(0, len(sliceProf90)): #Slice profile
            signal += cpmg(NEchoes, sliceProf90[curFAIndex]*B1Factor, sliceProf180[curFAIndex]*B1Factor, echoSpacing, T1, T2, False)
        signal /= len(sliceProf90)
        
        return np.abs(signal).astype(np.float)
    
    
    
    def getObjectiveFunction(yVector, T1):
        def objFun(params):
            a = params[0]
            t2 = params[1]
            b1 = params[2]
            
            plt.hold(False)
            plt.plot(yVector, 'bo')
            plt.hold(True)
                    
            signal = a * _signalCalc(T1, t2, b1)
            plt.plot(signal, 'r')
            plt.hold(False)
            plt.pause(0.001)
            
            return linalg.norm(signal  - yVector )
        
        return objFun # this is a closure with yVector as upvalue

    print("Fitting water")
    objFun = getObjectiveFunction(waterSignal, T1w)
    optParam = minimize(objFun, (waterSignal.max(), 40, 1), bounds = ((0,10000), t2Lim, b1Lim) ).x
    print(optParam)
    
    
#    print "Fitting fat"
#    objFun = getObjectiveFunction(fatSignal, T1f)
#    optParam = minimize(objFun, (fatSignal.max(), 40, 1), bounds = ((0,10000), t2Lim, b1Lim) ).x
#    print optParam