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
"""
from __future__ import print_function

import numpy as np
import os.path

from pycuda.autoinit import context
import pycuda.gpuarray as ga
from pycuda.compiler import SourceModule
import time

from FatFractionLookup import FatFractionLookup

CUDA_FILE = os.path.join( os.path.dirname(os.path.realpath(__file__)), "epg.cu")

def getCudaFunction(nEchoes, echoSpacing, T1f, T1w, magPrep = False):
    # prepare definitions
    definitions = """
        #define PYCUDA_COMPILE
        #define NECHOES {:d}
        #define ECHOSPACING {:.2f}f
        #define T1F {:.2f}f
        #define T1W {:.2f}f
        #define MAGPREP {:d}
        """.format( int(nEchoes), echoSpacing, T1f, T1w, magPrep)
    
    with open(CUDA_FILE, 'r') as cudaFile:
        source = cudaFile.read()
    
    # add specific definitions to source 
    source = definitions + source
    mod = SourceModule(source, no_extern_c=True)
    return mod.get_function("cpmg_sliceprof_B1_FF")

class FatFractionLookup_GPU(FatFractionLookup):
    
    T1w = 1400
    T1f = 365
    TBW = 2.0
#    NT2s = 200 # number of calculated T2 points
#    NB1s = 50 # number of calculated B1 points
    NT2s = 60 # number of calculated T2 points
    NB1s = 20 # number of calculated B1 points
    MagPreparePulse = False
    NFF = 101
    
    CudaBlockSize=256 # number of threads
    
    def __init__(self, T2Limits, B1Limits, FatT2, NEchoes, EchoSpacing, refWidthFactor = 0.2):
        FatFractionLookup.__init__(self, T2Limits, B1Limits, FatT2, NEchoes, EchoSpacing, refWidthFactor)
        self.allSignals = None
        self.parameterCombinations = None
    
    def generateSignals(self):
        starttime = time.time()
        parameterCombinations = []
        ffVector = np.linspace(0,1,self.NFF)
        for ffInd in range(len(ffVector)):
            for t2Ind in range(len(self.T2Points)):
                for b1Ind in range(len(self.B1Points)):
                    parameterCombinations.append( (self.T2Points[t2Ind], self.B1Points[b1Ind], ffVector[ffInd]) )
        
        parameterCombinations = np.array(parameterCombinations, dtype=np.float32)
        
        nParams = parameterCombinations.shape[0]
        
        signalsOut_gpu = ga.zeros( (nParams * self.NEchoes ), np.float32 )
        
        print("Compiling/loading CUDA module...")
        cuda_cpmg = getCudaFunction(self.NEchoes, self.EchoSpacing, self.T1f, self.T1w, self.MagPreparePulse)
        print("Generating signals...")
        
        params_gpu = ga.to_gpu(parameterCombinations.ravel())
        sp90_gpu = ga.to_gpu(self.sliceProf90.squeeze().astype(np.float32))
        sp180_gpu = ga.to_gpu(self.sliceProf180.squeeze().astype(np.float32))
        
        nBlocks = int(np.ceil( float(nParams) / self.CudaBlockSize ) )
        
        cuda_cpmg(np.uint32(nParams), np.uint32(self.sliceProf90.shape[0]), np.float32(self.fatT2), sp90_gpu, sp180_gpu, params_gpu, signalsOut_gpu, block=(self.CudaBlockSize,1,1), grid=(nBlocks,1))
        
        signalsOut_gpu = signalsOut_gpu.reshape( (nParams, self.NEchoes ) )
        
        context.synchronize()
        
        signalsOut = signalsOut_gpu.get()
        
        print("Done")
        print("Time taken:", time.time() - starttime)
        
        params_gpu.gpudata.free()
        sp90_gpu.gpudata.free()
        sp180_gpu.gpudata.free()
        
        self.allSignals = signalsOut
        self.parameterCombinations = parameterCombinations
        self.signalsReady = True
    
    def getAllSignals(self):
        if not self.signalsReady: self.generateSignals()
        return self.parameterCombinations, self.allSignals
    
    def getSignal(self, T2, B1, fatFraction):
        return NotImplementedError

if __name__ == '__main__':
    # test
    ff_gpu = FatFractionLookup_GPU( (20, 100), (0.6, 1.4), 151, 11, 10.9 )
    params, signals = ff_gpu.getAllSignals()
    
    print(params[123])
    print(signals[123,:])



