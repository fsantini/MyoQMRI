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
    
    Copyright 2020 Francesco Santini <francesco.santini@unibas.ch>
"""
from __future__ import print_function

import numpy as np

from pycuda.autoinit import context
import pycuda.gpuarray as ga
from pycuda.compiler import SourceModule
import os.path

CUDA_FILE = os.path.join( os.path.dirname(os.path.realpath(__file__)), "findmax_ff.cu")
CUDA_BLOCK_SIZE = 256

def getCudaFunction():
    with open(CUDA_FILE, 'r') as cudaFile:
        source = cudaFile.read()
    
    mod = SourceModule(source, no_extern_c=True)
    return mod.get_function("findmax_ff")

print("Findmax: Compiling/loading CUDA module...")
findmax_gpu_fn = getCudaFunction()
print("Done")

def prepareAndLoadParams(parameterCombinations):
    ffParams = np.round(parameterCombinations[:,2]*100).squeeze().astype(np.int32)
    return ga.to_gpu(ffParams)

def prepareAndLoadFF(ffValues):
    ffValues_conv = np.round(ffValues.ravel()*100).squeeze().astype(np.int32)
    return ga.to_gpu(ffValues_conv)
    
def findmax_gpu(corrMatrix_gpu, ffValues_gpu, ffParams_gpu):
    nVoxels = ffValues_gpu.shape[0]
    nParams = ffParams_gpu.shape[0]
    indexOut_gpu = ga.zeros((nVoxels), np.int32)
    nBlocks = int(np.ceil( float(nVoxels) / CUDA_BLOCK_SIZE ) )
    findmax_gpu_fn(corrMatrix_gpu, ffValues_gpu, np.uint32(nVoxels), ffParams_gpu, np.uint32(nParams), indexOut_gpu, block=(CUDA_BLOCK_SIZE,1,1), grid=(nBlocks,1))
    context.synchronize()
    indexOut_host = indexOut_gpu.get()
    indexOut_gpu.gpudata.free()
    return indexOut_host
    
    
    