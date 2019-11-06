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

from pycuda.autoinit import context
import pycuda.gpuarray as ga
from pycuda.compiler import SourceModule

from FatFractionLookup import calcSliceprof, reduceSliceProf, FatFractionLookup

CUDA_FILE = "epg.cu"

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
    TBW = 1.6
#    NT2s = 200 # number of calculated T2 points
#    NB1s = 50 # number of calculated B1 points
    NT2s = 60 # number of calculated T2 points
    NB1s = 20 # number of calculated B1 points
    MagPreparePulse = False
    NFF = 100
    
    CudaBlockSize=256 # number of threads
    
    def __init__(self, T2Limits, B1Limits, FatT2, NEchoes, EchoSpacing):
        self.fatT2 = FatT2
        self.NEchoes = NEchoes
        self.EchoSpacing = EchoSpacing
        self.fatSignals = None
        self.waterSignals = None
        self.sliceProf90 = reduceSliceProf(calcSliceprof(90, 2.0),10)
        self.sliceProf90 = np.pad(self.sliceProf90, ((0,2),(0,0)), 'constant')
        self.sliceProf180 = reduceSliceProf(calcSliceprof(180, 2.0),12) # maybe take into account larger slice for refocusing
        
        self.sliceProf90[np.isnan(self.sliceProf90)] = 0.0
        self.sliceProf180[np.isnan(self.sliceProf180)] = 0.0
        
        assert self.sliceProf90.shape == self.sliceProf180.shape, "Slice profiles for excitation and refocusing must be same"
        
        self.T2Points = np.linspace(T2Limits[0], T2Limits[1], self.NT2s)
        self.B1Points = np.linspace(B1Limits[0], B1Limits[1], self.NB1s)
    
    def getAllSignals(self):
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
        
        params_gpu.gpudata.free()
        sp90_gpu.gpudata.free()
        sp180_gpu.gpudata.free()
        
        return parameterCombinations, signalsOut
        
    def getSignal(T2, B1, fatFraction):
        raise NotImplementedError 

if __name__ == '__main__':
    # test
    ff_gpu = FatFractionLookup_GPU( (20, 100), (0.6, 1.4), 151, 11, 10.9 )
    params, signals = ff_gpu.getAllSignals()
    
    print(params[123])
    print(signals[123,:])


