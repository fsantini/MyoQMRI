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

from dicomUtils import load3dDicom, save3dDicom

import numpy as np
import numpy.matlib as matlib
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import os
import time
import gc
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser



INITIAL_FATT2 = 151

# DEFAULTS
NOISELEVEL = 300
fatT2 = INITIAL_FATT2 # From Marty #46 microlipds from paper?
NTHREADS = None
DOPLOT=0
t2Lim = (20,80)
#t2Lim = (50,600)
b1Lim = (0.4,1.4)

parser = ArgumentParser(description='Fit a multiecho dataset')
parser.add_argument('path', type=str, help='path to the dataset')
parser.add_argument('--fat-t2', '-f', metavar='T2', dest='fatT2', type=float, help='fat T2 (default: %.f)' % (fatT2), default = fatT2)
parser.add_argument('--noise-level', '-n', dest='noiselevel', metavar='N', type=int, help='noise level for thresholding (default: %d)' % (NOISELEVEL), default = NOISELEVEL)
parser.add_argument('--nthreads', '-t', dest='nthreads', metavar='T', type=int, help='number of threads to be used for fitting (default: %d)' % cpu_count(), default = cpu_count())
parser.add_argument('--plot-level', '-p', metavar='L', dest='doplot', type=int, help='do a live plot of the fitting (L=0: no plot, L=1: show the images, L=2: show images and signals)', default=DOPLOT)
parser.add_argument('--t2-limits', metavar=('min', 'max'), dest='t2Lim', type=int, nargs=2, help='set the limits for t2 calculation (default: %d-%d)' % t2Lim, default = t2Lim)
parser.add_argument('--b1-limits', metavar=('min', 'max'), dest='b1Lim', type=float, nargs=2, help='set the limits for b1 calculation (default: %.1f-%.1f)' % b1Lim, default = b1Lim)
parser.add_argument('--use-gpu', '-g', dest='useGPU', metavar='TF', type=int, help='use GPU for fitting (0 == false)', default=0)

args = parser.parse_args()

NOISELEVEL = args.noiselevel
fatT2 = args.fatT2
baseDir = args.path
NTHREADS = args.nthreads
DOPLOT = args.doplot
t2Lim = args.t2Lim
b1Lim = args.b1Lim
useGPU = args.useGPU

if useGPU != 0:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import skcuda.linalg as sklinalg
    import skcuda.misc as skmisc
    from FatFractionLookup_GPU import FatFractionLookup_GPU as FatFractionLookup
    
    skmisc.init()
    NTHREADS = 1
else:
    from FatFractionLookup import FatFractionLookup
    
print("Base dir:", baseDir)
print("NOISELEVEL:", NOISELEVEL)
print("Fat T2:", fatT2)
print("N Threads:", NTHREADS)
print("PLot level:", DOPLOT)
print("T2 limits", t2Lim)
print("B1 limits", b1Lim)

[dicomStack, infos] = load3dDicom(baseDir)

etl = int(infos[0].EchoTrainLength)
echoSpacing = float(infos[0].EchoTime)

print("Echo Train Length:", etl)
print("Echo spacing:", echoSpacing)

oldShape = dicomStack.shape
newShape = (oldShape[0], oldShape[1], etl, int(oldShape[2]/etl))

print(newShape)

nSlices = newShape[3] 

dicomStack = dicomStack.reshape(newShape).swapaxes(2,3) # reorder as slice, etl instead of etl, slices

newShape = dicomStack.shape

infoOut = infos[:nSlices]

plt.ion()

ffl = None

if fatT2 <= 0:
    ffl = FatFractionLookup(t2Lim, b1Lim, INITIAL_FATT2, etl, echoSpacing)
else:
    ffl = FatFractionLookup(t2Lim, b1Lim, fatT2, etl, echoSpacing)
    
parameterCombinations, signals = ffl.getAllSignals()
signals = signals ** 2 # weight by magnitude
signorms = linalg.norm(signals, axis=1, keepdims=True)
signormsRep = np.repeat(signorms, signals.shape[1], axis=1)
signalsNormalized = signals/signormsRep


def findBestMatch(signal):
    global DOPLOT
    #signal /= signal[0]
    #signalMatrix = matlib.repmat(signal ** 2, len(parameterCombinations),1)

    #n = np.sum( (signalMatrix - signals) ** 2, axis = 1 ) #linalg.norm(signalMatrix - signals, axis = 1)
    #nIndex = np.argmin(n)
    
    
    signal = signal**2
    #signal /= signal[0]
    #print(signal)
    
    n = np.dot(signalsNormalized, signal)
    nIndex = np.argmax(n)
    
    if DOPLOT >= 2:
        plt.figure("SigPlot")
        plt.plot(signal)
        plt.hold(True)
        plt.plot(signals[np.argmin(n), :], 'rd')
        plt.hold(False)
        plt.title("t2: %.1f, b1: %.1f, ff: %.1f".format(parameterCombinations[nIndex]))
        plt.pause(0.001)
    
    return parameterCombinations[nIndex]
                    
         
def getFindBestMatchLocal(pComb, dictionary):
    dictionaryLocal = np.copy(dictionary)
    def findBestMatchLocal(signal):
        signal /= signal[0]
        signalMatrix = matlib.repmat(signal**2, len(pComb),1)
        n = np.sum( (signalMatrix - dictionaryLocal) ** 2, axis = 1 ) #linalg.norm(signalMatrix - signals, axis = 1)
        return pComb[np.argmin(n)]
    return findBestMatchLocal

def fitSlcMultiprocess(slcData, srcFatT2, t2b1ff, findBestMatchLocal):
    fatSignal = 0
    nFatSignals = 0
    sz = slcData.shape
    for i in range(sz[0]):
        for j in range(sz[1]):
            yValues = slcData[i, j, :].squeeze()
            if yValues.max() < NOISELEVEL: continue                
            optParam = findBestMatchLocal(yValues)
            t2_val = optParam[0]
            b1_val = optParam[1]
            ff_val = optParam[2]
            if srcFatT2:
                if ff_val > 0.8:
                    fatSignal += yValues
                    nFatSignals += 1
                if nFatSignals > 20:
                    t2, b1 = ffl.cpmgFit(fatSignal, ffl.T1f)
                    print("Calculated fat T2:", t2, "b1:", b1)
                    return t2
            else:       
                t2b1ff[0,i,j] = optParam[0]
                t2b1ff[1,i,j] = optParam[1]
                t2b1ff[2,i,j] = optParam[2]

def fitMultiProcess(slcData):
    findBestMatchLocal = getFindBestMatchLocal(parameterCombinations, signals)
    sz = slcData.shape
    t2b1ff = np.zeros( (3, sz[0], sz[1]) )                
    if fatT2 <= 0:
       print("Searching fat...")
       localfatT2 = fitSlcMultiprocess(slcData, True, t2b1ff, findBestMatchLocal)
       if localfatT2 is None:
           return t2b1ff
       localFfl = FatFractionLookup(t2Lim, b1Lim, localfatT2, etl, echoSpacing)
       localPars, localSigs = localFfl.getAllSignals()
       localSigs = localSigs ** 2 # weight by magnitude
       findBestMatchLocal = getFindBestMatchLocal(localPars, localSigs)
       
    fitSlcMultiprocess(slcData, False, t2b1ff, findBestMatchLocal)
    print("Exiting fitMultiProcess")
    return t2b1ff

outShape = newShape[0:3]    

t = time.time()

# multiprocess fitting
if NTHREADS != 1:
    if NTHREADS:
        p = Pool(NTHREADS)
    else:
        p = Pool() # automatic number of processes
    
    #print fitMultiProcess(dicomStack[:,:,1,:])
    
    resultList = np.array(p.map(fitMultiProcess, dicomStack))
    #resultList = np.array(p.map(fitMultiProcess, dicomStack)) # no processes
    # remap list
    t2 = resultList[:,0,:,:].squeeze()
    b1 = resultList[:,1,:,:].squeeze()
    ff = resultList[:,2,:,:].squeeze()
            
    p.close()
    p.join()
    
else:

# single-process fitting
    t2 = np.zeros(outShape)
    b1 = np.zeros(outShape)
    ff = np.zeros(outShape)
    
    if useGPU:
        signorms = linalg.norm(signals, axis=1, keepdims=True)
        signormsRep = np.repeat(signorms, signals.shape[1], axis=1)
        signormsGPU = pycuda.gpuarray.to_gpu(signormsRep.astype(np.float32))
        signalsGPU = pycuda.gpuarray.to_gpu(signals.astype(np.float32))
        signalsGPU = sklinalg.transpose(skmisc.divide(signalsGPU, signormsGPU))
        del signormsGPU
        ROWSTEP = 14
    
    def fitSlcGPU(slc, srcFatT2, t2, b1, ff):
        global ROWSTEP
        print("Fitting slice", slc)
        yValues = dicomStack[:, :, slc, :].squeeze()
        slcShape = yValues.shape
        nrows = slcShape[0]
        ncols = slcShape[1]
        sigLen = slcShape[2]
        success = False
        
        while not success:
            try:
                for r in range(0,nrows,ROWSTEP):
                    rowMax = min(r+ROWSTEP, nrows)
                    #print r
                    slcLin = yValues[r:rowMax,:,:].reshape(ncols*(rowMax-r), sigLen).astype(np.float32)
                    
                    slcGPU = pycuda.gpuarray.to_gpu(slcLin)
                    slcGPU = sklinalg.multiply(slcGPU, slcGPU)
                    
                    
                    #print slcGPU.shape
                    #print signalsGPU.shape
                    
                    corrMatrixGPU = sklinalg.mdot(slcGPU, signalsGPU) # correlation
                    
                    
                    corrMaxGPU = skmisc.argmax(corrMatrixGPU, 1)
                    #print corrMaxGPU.shape
                    corrMax = corrMaxGPU.get()
                    for row in range(r, rowMax):
                        for c in range(ncols):
                            ind = (row-r)*ncols + c
                            #print ind
                            t2[row,c,slc] = parameterCombinations[corrMax[ind]][0]
                            b1[row,c,slc] = parameterCombinations[corrMax[ind]][1]
                            ff[row,c,slc] = parameterCombinations[corrMax[ind]][2]
                            
                    #show images
                    if DOPLOT >= 1:
                        plt.figure("ImaPlot")
                        plt.suptitle("Slice %d of %d" % (slc+1, newShape[2]))
                        plt.subplot(131)
                        plt.imshow(t2[:,:,slc])
                        plt.axis('image')
                        frame1 = plt.gca()
                        frame1.axes.get_xaxis().set_visible(False)
                        frame1.axes.get_yaxis().set_visible(False)
                        plt.title("T2")
                        plt.subplot(132)
                        plt.imshow(b1[:,:,slc])
                        plt.axis('image')
                        frame1 = plt.gca()
                        frame1.axes.get_xaxis().set_visible(False)
                        frame1.axes.get_yaxis().set_visible(False)
                        plt.title("B1")
                        plt.subplot(133)
                        plt.imshow(ff[:,:,slc])
                        plt.axis('image')
                        frame1 = plt.gca()
                        frame1.axes.get_xaxis().set_visible(False)
                        frame1.axes.get_yaxis().set_visible(False)
                        plt.title("FF")
                        plt.pause(0.001)
                        
                success = True
            except pycuda._driver.MemoryError:
                ROWSTEP -= 1
                gc.collect()
                print("Not enough GPU Mem: decreasing ROWSTEP to", ROWSTEP)
                
        
    signorms = linalg.norm(signals, axis=1, keepdims=True)
    signormsRep = np.repeat(signorms, signals.shape[1], axis=1)
    signalsCPU = np.transpose( signals / signormsRep)
    ROWSTEP = 14
    
    def fitSlcFast(slc, srcFatT2, t2, b1, ff):
        global ROWSTEP
        print("Fitting slice", slc)
        yValues = dicomStack[:, :, slc, :].squeeze()
        slcShape = yValues.shape
        nrows = slcShape[0]
        ncols = slcShape[1]
        sigLen = slcShape[2]
        
        for r in range(0,nrows,ROWSTEP):
            rowMax = min(r+ROWSTEP, nrows)
            #print r
            slcCPU = yValues[r:rowMax,:,:].reshape(ncols*(rowMax-r), sigLen)
            
            slcCPU = slcCPU * slcCPU
            
            
            #print slcGPU.shape
            #print signalsGPU.shape
            
            corrMatrixCPU = np.dot(slcCPU, signalsCPU) # correlation
            
            corrMax = np.argmax(corrMatrixCPU, 1)
            #print corrMaxGPU.shape
            for row in range(r, rowMax):
                for c in range(ncols):
                    ind = (row-r)*ncols + c
                    #print ind
                    t2[row,c,slc] = parameterCombinations[corrMax[ind]][0]
                    b1[row,c,slc] = parameterCombinations[corrMax[ind]][1]
                    ff[row,c,slc] = parameterCombinations[corrMax[ind]][2]
                    
            #show images
            if DOPLOT >= 1:
                plt.figure("ImaPlot")
                plt.suptitle("Slice %d of %d" % (slc+1, newShape[2]))
                plt.subplot(131)
                plt.imshow(t2[:,:,slc])
                plt.axis('image')
                frame1 = plt.gca()
                frame1.axes.get_xaxis().set_visible(False)
                frame1.axes.get_yaxis().set_visible(False)
                plt.title("T2")
                plt.subplot(132)
                plt.imshow(b1[:,:,slc])
                plt.axis('image')
                frame1 = plt.gca()
                frame1.axes.get_xaxis().set_visible(False)
                frame1.axes.get_yaxis().set_visible(False)
                plt.title("B1")
                plt.subplot(133)
                plt.imshow(ff[:,:,slc])
                plt.axis('image')
                frame1 = plt.gca()
                frame1.axes.get_xaxis().set_visible(False)
                frame1.axes.get_yaxis().set_visible(False)
                plt.title("FF")
                plt.pause(0.001)

    
    
    def fitSlc(slc, srcFatT2, t2, b1, ff):
        print("Fitting slice", slc)
        fatSignal = 0
        nFatSignals = 0
        for col in range(newShape[1]):
            for row in range(newShape[0]):
                yValues = dicomStack[row, col, slc, :].squeeze()
                if yValues.max() < NOISELEVEL: continue                
                optParam = findBestMatch(yValues)
                t2_val = optParam[0]
                b1_val = optParam[1]
                ff_val = optParam[2]
                if srcFatT2:
                    if ff_val > 0.9:
                        print(t2_val, ff_val, b1_val)
                        fatSignal += yValues
                        nFatSignals += 1
                    if nFatSignals > 40:
                        t2, b1 = ffl.cpmgFit(fatSignal, ffl.T1f)
                        print("Calculated fat T2:", t2, "b1:", b1)
                        return t2
                else:       

                    t2[row,col,slc] = optParam[0]
                    b1[row,col,slc] = optParam[1]
                    ff[row,col,slc] = optParam[2]
                                
                    #show images
                    if DOPLOT >= 1:
                        plt.figure("ImaPlot")
                        plt.suptitle("Slice %d of %d" % (slc+1, newShape[2]))
                        plt.subplot(131)
                        plt.imshow(t2[:,:,slc])
                        plt.axis('image')
                        frame1 = plt.gca()
                        frame1.axes.get_xaxis().set_visible(False)
                        frame1.axes.get_yaxis().set_visible(False)
                        plt.title("T2")
                        plt.subplot(132)
                        plt.imshow(b1[:,:,slc])
                        plt.axis('image')
                        frame1 = plt.gca()
                        frame1.axes.get_xaxis().set_visible(False)
                        frame1.axes.get_yaxis().set_visible(False)
                        plt.title("B1")
                        plt.subplot(133)
                        plt.imshow(ff[:,:,slc])
                        plt.axis('image')
                        frame1 = plt.gca()
                        frame1.axes.get_xaxis().set_visible(False)
                        frame1.axes.get_yaxis().set_visible(False)
                        plt.title("FF")
                        plt.pause(0.001)
        
    for slc in range(newShape[2]):
       if fatT2 <= 0:
           print("Searching fat...")
           fatT2 = fitSlc(slc, True, t2, b1, ff)
           ffl = FatFractionLookup(t2Lim, b1Lim, fatT2, etl, echoSpacing)
           parameterCombinations, signals = ffl.getAllSignals()
           signals = signals ** 2 # weight by magnitude
          
       if useGPU:
           fitSlcGPU(slc, False, t2, b1, ff)
       else:
           fitSlcFast(slc, False, t2, b1, ff)
       



print("Elapsed time", time.time() - t)

save3dDicom(t2*10, infoOut, os.path.join(baseDir, 't2'), 97)
save3dDicom(b1*100, infoOut, os.path.join(baseDir, 'b1'), 98)
save3dDicom(ff*100, infoOut, os.path.join(baseDir, 'ff'), 99)
    
