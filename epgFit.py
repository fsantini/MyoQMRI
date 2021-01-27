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
import scipy.optimize as opt

INITIAL_FATT2 = 151

# DEFAULTS
NOISELEVEL = 300
fatT2 = INITIAL_FATT2 # From Marty #46 microlipids from paper?
NTHREADS = None
DOPLOT=0
t2Lim = (20,80)
#t2Lim = (50,600)
b1Lim = (0.5,1.2)
refocusingFactor = 1.2

parser = ArgumentParser(description='Fit a multiecho dataset')
parser.add_argument('path', type=str, help='path to the dataset')
parser.add_argument('--fit-type', '-y', metavar='T', dest='fitType', type=int, help='type of fitting: T=0: EPG, T=1: Single exponential, T=2: Double exponential (default: 0)', default=0)
parser.add_argument('--fat-t2', '-f', metavar='T2', dest='fatT2', type=float, help=f'fat T2 (default: {fatT2:.0f})', default = fatT2)
parser.add_argument('--noise-level', '-n', dest='noiselevel', metavar='N', type=int, help=f'noise level for thresholding (default: {NOISELEVEL})', default = NOISELEVEL)
parser.add_argument('--nthreads', '-t', dest='nthreads', metavar='T', type=int, help=f'number of threads to be used for fitting (default: {cpu_count()})', default = cpu_count())
parser.add_argument('--plot-level', '-p', metavar='L', dest='doplot', type=int, help='do a live plot of the fitting (L=0: no plot, L=1: show the images, L=2: show images and signals)', default=DOPLOT)
parser.add_argument('--t2-limits', metavar=('min', 'max'), dest='t2Lim', type=int, nargs=2, help=f'set the limits for t2 calculation (default: {t2Lim[0]}-{t2Lim[1]})', default = t2Lim)
parser.add_argument('--b1-limits', metavar=('min', 'max'), dest='b1Lim', type=float, nargs=2, help=f'set the limits for b1 calculation (default: {b1Lim[0]:.1f}-{b1Lim[1]:.1f})', default = b1Lim)
parser.add_argument('--use-gpu', '-g', dest='useGPU',action='store_true', help='use GPU for fitting')
parser.add_argument('--ff-map', '-m', metavar='dir', dest='ffMapDir', type=str, help='load a fat fraction map', default='')
parser.add_argument('--register-ff', '-r', dest='regFF', action='store_true', help='register the fat fraction dataset')
parser.add_argument('--etl-limit', '-e', metavar='N', dest='etlLimit', type=int, help='reduce the echo train length', default=0)
parser.add_argument('--out-suffix', '-s', metavar='ext', dest='outSuffix', type=str, help='add a suffix to the output map directories', default='')
parser.add_argument('--slice-range', '-l', metavar=('start', 'end'), dest='sliceRange', type=int, nargs=2, help='Restrict the fitting to a subset of slices', default=(None, None))
parser.add_argument('--refocusing-width', '-w', metavar='factor', dest='refocusingFactor', type=float, help=f'Slice width of the refocusing pulse with respect to the excitation (default {refocusingFactor}) (Siemens standard)', default=refocusingFactor)
parser.add_argument('--exc-profile', metavar='path', dest='excProfilePath', type=str, help='Path to the excitation slice profile file', default=None)
parser.add_argument('--ref-profile', metavar='path', dest='refProfilePath', type=str, help='Path to the refocusing slice profile file', default=None)



args = parser.parse_args()


NOISELEVEL = args.noiselevel
fatT2 = args.fatT2
baseDir = args.path
NTHREADS = args.nthreads
DOPLOT = args.doplot
t2Lim = args.t2Lim
b1Lim = args.b1Lim
useGPU = args.useGPU
ffMapDir = args.ffMapDir
etlLimit = args.etlLimit
regFF = args.regFF
outSuffix = args.outSuffix
fitType = args.fitType
sliceRange = args.sliceRange
refocusingFactor = args.refocusingFactor
excProfilePath = args.excProfilePath
refProfilePath = args.refProfilePath

print("Base dir:", baseDir)
print("NOISELEVEL:", NOISELEVEL)
print("Fit type:", fitType)
print("Fat T2:", fatT2)
print("N Threads:", NTHREADS)
print("PLot level:", DOPLOT)
print("T2 limits", t2Lim)
print("B1 limits", b1Lim)
print("Use GPU", useGPU)
print("FF Map Dir", ffMapDir)
print("Reg FF", regFF)
print("ETL limit", etlLimit)
print("Output suffix", outSuffix)
print("Slice Range", sliceRange)
print("Refocusing Factor", refocusingFactor)
print("Excitation slice profile", excProfilePath)
print("Refocusing slice profile", refProfilePath)

refocusingFactor -= 1.0 # the actual parameter passed must be 0.2

assert useGPU or ffMapDir == '' or NTHREADS == 1, "FF map can only be used with a single thread"
assert NTHREADS == 1 or fitType == 0, "Only EPG fitting can be used with multiple threads"
assert not useGPU or fitType == 0, "Only EPG fitting is supported on the GPU"
assert (excProfilePath is None and excProfilePath is None) or (excProfilePath is not None and excProfilePath is not None), "Either both slice profiles are specified, or neither is"

excProfile = None
refProfile = None

if excProfilePath:
    excProfile = np.loadtxt(excProfilePath)

if refProfilePath:
    refProfile = np.loadtxt(refProfilePath)
    
assert excProfile.shape == refProfile.shape and excProfile.ndim == 1, "Slice profiles must be one-dimensional vectors and contain the same number of samples"


###########################################################
## Initialization
###########################################################    

if useGPU:
    import pycuda.driver as cuda
    import pycuda.autoinit
    import skcuda.linalg as sklinalg
    import skcuda.misc as skmisc
    from FatFractionLookup_GPU import FatFractionLookup_GPU as FatFractionLookup
    import findmax_ff
    
    skmisc.init()
    NTHREADS = 1
else:
    from FatFractionLookup import FatFractionLookup
    
[dicomStack, infos] = load3dDicom(baseDir)

etl = int(infos[0].EchoTrainLength)
echoSpacing = float(infos[0].EchoTime)

oldShape = dicomStack.shape
newShape = (oldShape[0], oldShape[1], etl, int(oldShape[2]/etl))

print(newShape)

nSlices = newShape[3] 

if not any(sliceRange): sliceRange = (0, nSlices)

assert sliceRange[0] >= 0 and sliceRange[1] <= nSlices, "Selected slice range is out of bound"

dicomStack = dicomStack.reshape(newShape).swapaxes(2,3) # reorder as slice, etl instead of etl, slices

if etlLimit > 0 and etlLimit < etl:
    dicomStack = dicomStack[:,:,:,:etlLimit]
    etl = etlLimit

print("Echo Train Length:", etl)
print("Echo spacing:", echoSpacing)

newShape = dicomStack.shape

infoOut = infos[:nSlices]

plt.ion()

ffl = None

if fatT2 <= 0:
    ffl = FatFractionLookup(t2Lim, b1Lim, INITIAL_FATT2, etl, echoSpacing, refocusingFactor)
    if excProfile is not None: ffl.setPulsesExt(excProfile, refProfile, refocusingFactor)
else:
    ffl = FatFractionLookup(t2Lim, b1Lim, fatT2, etl, echoSpacing, refocusingFactor)
    if excProfile is not None: ffl.setPulsesExt(excProfile, refProfile, refocusingFactor)
    
if fitType == 0:
    parameterCombinations, signals = ffl.getAllSignals()
    signals = signals **2 # weight by magnitude
    #print("Signals are Nan", np.any(np.isnan(signals)))
    signorms = linalg.norm(signals, axis=1, keepdims=True)
    signormsRep = np.repeat(signorms, signals.shape[1], axis=1)
    signalsNormalized = signals/signormsRep
    #print("Signal Norm  are nan", np.any(np.isnan(signalsNormalized)))

signalsFF = None
parameterCombinationsFF = None


###########################################################
## Standard fitting
###########################################################    

def findBestMatchFF(signal, fatfraction_in):
    global DOPLOT, signalsFF, parameterCombinationsFF
    if not signalsFF:
        print(parameterCombinations.shape)
        print(signalsNormalized.shape)
        parameterCombinationsFF = []
        signalsFF = []
        # precalculate the signals divided by FF
        for ff in range(0,101):
            indices = np.where(np.round(parameterCombinations[:,2]*100).astype(np.int16) == ff)
            parameterCombinationsFF.append(parameterCombinations[indices,:].squeeze())
            signalsFF.append(signalsNormalized[indices,:].squeeze())
    
    ff = int(round(fatfraction_in*100))
    if ff < 0: ff = 0
    if ff > 100: ff = 100
    
    signal = signal**2
    
    n = np.dot(signalsFF[ff], signal)
    nIndex = np.argmax(n)
    
    if DOPLOT >= 2:
        plotSignals(signal, signalsFF[ff][nIndex, :], parameterCombinationsFF[ff][nIndex,:])
    
    return parameterCombinationsFF[ff][nIndex]
    

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
        plotSignals(signal, signals[nIndex, :], parameterCombinations[nIndex,:])
    
    return parameterCombinations[nIndex]
        

def fitSlcFast(slc, srcFatT2, t2, b1, ff):
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
            plotImages()
            
def fitSlc(slc, srcFatT2, t2, b1, ff):
    print("Fitting slice", slc)
    fatSignal = 0
    nFatSignals = 0
    try:
        useFF = True if np.any(ff[:,:,slc] > 0) else False
    except:
        print("Warning while checking useFF")
        useFF = False
    for col in range(newShape[1]):
        for row in range(newShape[0]):
            yValues = dicomStack[row, col, slc, :].squeeze()
            if yValues.max() < NOISELEVEL: continue 
            if useFF:
                optParam = findBestMatchFF(yValues, ff[row,col,slc])
            else:
                optParam = findBestMatch(yValues)
            t2_val = optParam[0]
            b1_val = optParam[1]
            ff_val = optParam[2]
            if srcFatT2:
                if ff_val > 0.9:
                    print(t2_val, ff_val, b1_val)
                    fatSignal += yValues
                    nFatSignals += 1
                if nFatSignals > 10:
                    t2, b1 = ffl.cpmgFit(fatSignal, ffl.T1f)
                    print("Calculated fat T2:", t2, "b1:", b1)
                    return t2
            else:       

                t2[row,col,slc] = optParam[0]
                b1[row,col,slc] = optParam[1]
                ff[row,col,slc] = optParam[2]
                            
                #show images
                if DOPLOT >= 1:
                    plotImages()


###########################################################
## Multiprocess
###########################################################            
         
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
       localFfl = FatFractionLookup(t2Lim, b1Lim, localfatT2, etl, echoSpacing, refocusingFactor)
       if excProfile is not None: ffl.setPulsesExt(excProfile, refProfile, refocusingFactor)
       localPars, localSigs = localFfl.getAllSignals()
       localSigs = localSigs **2 # weight by magnitude
       findBestMatchLocal = getFindBestMatchLocal(localPars, localSigs)
       
    fitSlcMultiprocess(slcData, False, t2b1ff, findBestMatchLocal)
    print("Exiting fitMultiProcess")
    return t2b1ff

###################################################################
## Non linear fitting
###################################################################

def singleExpFit(xValues, yValues, ffValue = None):
    def t2Model(te,a,t2):
        return a*np.exp(-te/t2)
    
    # remove first echo
    xValues = xValues[1:]
    yValues = yValues[1:]
    
    optParam, cov = opt.curve_fit(t2Model, xValues, yValues, p0 = (yValues[0], 50), bounds = ([0,10], [np.inf, 200]) )
    b1_val = 1    
    t2_val = optParam[1]
    ff_val = 0
    if DOPLOT >= 2:
        plotSignals(yValues, t2Model(xValues, *optParam), (t2_val, b1_val, ff_val))
    return b1_val, t2_val, ff_val

def doubleExpFit(xValues, yValues, ffValue = None):
    def doublet2Model(te,a,t2,ff):
        return a*(1-ff)*np.exp(-te/t2) + a*ff*np.exp(-te/fatT2)
    
    def doublet2ModelFixFF(te,a,t2):
        return a*(1-ffValue)*np.exp(-te/t2) + a*ffValue*np.exp(-te/fatT2)
    
    # remove first echo
    xValues = xValues[1:]
    yValues = yValues[1:]
    
    if ffValue is None:
        try:
            optParam, cov = opt.curve_fit(doublet2Model, xValues, yValues, p0 = (yValues[0], 50, 0), bounds = ([0,10,0], [np.inf, 100, 1]) )
        except:
            optParam = [0,1,0]
        b1_val = 1    
        t2_val = optParam[1]
        ff_val = optParam[2]
        if DOPLOT >= 2:
            plotSignals(yValues, doublet2Model(xValues, *optParam), (t2_val, b1_val, ff_val))
    else:
        try:
            optParam, cov = opt.curve_fit(doublet2ModelFixFF, xValues, yValues, p0 = (yValues[0], 50), bounds = ([0,10], [np.inf, 100]) )
        except:
            optParam = [0,1]
        b1_val = 1    
        t2_val = optParam[1]
        ff_val = ffValue
        if DOPLOT >= 2:
            plotSignals(yValues, doublet2ModelFixFF(xValues, *optParam), (t2_val, b1_val, ff_val))
    return b1_val, t2_val, ff_val

def fitSlcNLin(slc, fitFunction, t2, b1, ff):
    print("Fitting slice", slc)
    useFF = True if np.any(ff[:,:,slc] > 0) else False
    teVec = np.arange(1,etl+1)*echoSpacing
    
    for col in range(newShape[1]):
        for row in range(newShape[0]):
            yValues = dicomStack[row, col, slc, :].squeeze()
            if yValues.max() < NOISELEVEL: continue 
            
            
            if useFF:
                b1_val, t2_val, ff_val = fitFunction(teVec, yValues, ff[row,col,slc])
            else:
                b1_val, t2_val, ff_val = fitFunction(teVec, yValues, None)
            
            t2[row,col,slc] = t2_val
            b1[row,col,slc] = b1_val
            ff[row,col,slc] = ff_val

        #show images
        if DOPLOT >= 1:
            plotImages()    


###################################################################
## GPU Functions
###################################################################

def tryFree(gpuarr):
    try:
        gpuarr.gpudata.free()
    except:
        pass

def fitSlcGPU(slc, srcFatT2, t2, b1, ff):
    global ROWSTEP
    print("Fitting slice", slc)
    yValues = dicomStack[:, :, slc, :].squeeze()
    slcShape = yValues.shape
    nrows = slcShape[0]
    ncols = slcShape[1]
    sigLen = slcShape[2]
    success = False
    
    ffParams_gpu = None
    ffValues_gpu = None
    
    if np.any(ff[:,:,slc] > 0):
        useFF = True
        ffParams_gpu = findmax_ff.prepareAndLoadParams(parameterCombinations)
    else:
        useFF = False
        
    while not success:
        try:
            for r in range(0,nrows,ROWSTEP):
                rowMax = min(r+ROWSTEP, nrows)
                slcLin = yValues[r:rowMax,:,:].reshape(ncols*(rowMax-r), sigLen).astype(np.float32)
                
                slcGPU = None
                
                slcGPU = pycuda.gpuarray.to_gpu(slcLin)
                slcGPU = sklinalg.multiply(slcGPU, slcGPU)
                corrMatrixGPU = sklinalg.mdot(slcGPU, signalsGPU) # correlation
                
                tryFree(slcGPU)
                
                if useFF:
                    ffValues_gpu = findmax_ff.prepareAndLoadFF(ff[r:rowMax, :, slc])
                    corrMax = findmax_ff.findmax_gpu(corrMatrixGPU, ffValues_gpu, ffParams_gpu)
                else:
                    corrMaxGPU = skmisc.argmax(corrMatrixGPU, 1)
                    corrMax = corrMaxGPU.get()
                    tryFree(corrMaxGPU)
                    
                tryFree(corrMatrixGPU)
                tryFree(ffValues_gpu)
                
                for row in range(r, rowMax):
                    for c in range(ncols):
                        ind = (row-r)*ncols + c
                        t2[row,c,slc] = parameterCombinations[corrMax[ind]][0]
                        b1[row,c,slc] = parameterCombinations[corrMax[ind]][1]
                        ff[row,c,slc] = parameterCombinations[corrMax[ind]][2]
                        
                if DOPLOT >= 1:
                    plotImages()
                    
            success = True
        except pycuda._driver.MemoryError:
            ROWSTEP -= 1
            tryFree(slcGPU)
            tryFree(corrMatrixGPU)
            tryFree(ffValues_gpu)
            
            gc.collect()
            print("Not enough GPU Mem: decreasing ROWSTEP to", ROWSTEP)

############################################################
## Plotting functions
############################################################

def plotSignals(realSignal, simSignal, t2b1ff):
    plt.figure("SigPlot")
    plt.clf()
    print(realSignal, simSignal)
    plt.plot(realSignal)
    plt.plot(realSignal[0]*simSignal/simSignal[0], 'rd')
    plt.title("t2: {:.1f}, b1: {:.1f}, ff: {:.1f}".format(t2b1ff[0], t2b1ff[1], t2b1ff[2]))
    plt.pause(0.001)

def plotImages():
    plt.figure("ImaPlot")
    plt.clf()
    plt.suptitle(f"Slice {slc+1} of {newShape[2]}")
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
    plt.imshow(ff[:,:,slc], vmin=0, vmax=1)
    plt.axis('image')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.title("FF")
    plt.pause(0.001)


## Main program

outShape = newShape[0:3]    

t = time.time()


# multiprocess fitting
if NTHREADS != 1:
    if NTHREADS:
        p = Pool(NTHREADS)
    else:
        p = Pool() # automatic number of processes
    
    dicomStack2 = np.zeros_like(dicomStack)
    dicomStack2[:,:,slice(*sliceRange),:] = dicomStack[:,:,slice(*sliceRange),:]
    
    resultList = np.array(p.map(fitMultiProcess, dicomStack2))
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
    if ffMapDir:
        ff, ffInfo = load3dDicom(ffMapDir)
        
        # registration of the ff dataset
        if not regFF and ff.shape != dicomStack[:,:,:,0].squeeze().shape:
            print("Fat Fraction and T2 datasets have different shapes. Registration forced")
            regFF = True
        if regFF:
            from registerDatasets import calcTransform2DStack
            print("Registering the FF dataset")
            transf = calcTransform2DStack(dicomStack[:,:,:,0], infoOut, ff, ffInfo)
            ff = transf(ff)

        ff[ff<0] = 0
        ff[ff>2**15] = 0 # sometimes there is a problem with saving signed/unsigned ff values
        while ff.max() > 7: # rescale ff
            ff /= 10
        # print(ff.max())
    else:
        ff = np.zeros(outShape)
    
    if useGPU:
        signorms = linalg.norm(signals, axis=1, keepdims=True)
        signormsRep = np.repeat(signorms, signals.shape[1], axis=1)
        signormsGPU = pycuda.gpuarray.to_gpu(signormsRep.astype(np.float32))
        signalsGPU = pycuda.gpuarray.to_gpu(signals.astype(np.float32))
        signalsGPU = sklinalg.transpose(skmisc.divide(signalsGPU, signormsGPU))
        del signormsGPU
        ROWSTEP = 14
                
    if fitType == 0:
        signorms = linalg.norm(signals, axis=1, keepdims=True)
        signormsRep = np.repeat(signorms, signals.shape[1], axis=1)
        signalsCPU = np.transpose( signals / signormsRep)
        ROWSTEP = 14
        
    for slc in range(*sliceRange):
        print(slc)
        if fatT2 <= 0:            
            print("Searching fat...")
            fatT2 = fitSlc(int((sliceRange[1]-sliceRange[0])/2+sliceRange[0]), True, t2, b1, ff)
            ffl = FatFractionLookup(t2Lim, b1Lim, fatT2, etl, echoSpacing, refocusingFactor)
            if excProfile is not None: ffl.setPulsesExt(excProfile, refProfile, refocusingFactor)
            parameterCombinations, signals = ffl.getAllSignals()
            signals = signals **2 # weight by magnitude
            signorms = linalg.norm(signals, axis=1, keepdims=True)
            signormsRep = np.repeat(signorms, signals.shape[1], axis=1)
            signalsNormalized = signals/signormsRep
            if useGPU:
                signorms = linalg.norm(signals, axis=1, keepdims=True)
                signormsRep = np.repeat(signorms, signals.shape[1], axis=1)
                signormsGPU = pycuda.gpuarray.to_gpu(signormsRep.astype(np.float32))
                signalsGPU = pycuda.gpuarray.to_gpu(signals.astype(np.float32))
                signalsGPU = sklinalg.transpose(skmisc.divide(signalsGPU, signormsGPU))
                del signormsGPU
          
        if useGPU:
            fitSlcGPU(slc, False, t2, b1, ff)
        else:
            if fitType == 1:
                fitSlcNLin(slc, singleExpFit, t2, b1, ff)
            elif fitType == 2:
                fitSlcNLin(slc, doubleExpFit, t2, b1, ff)
            else:
                if ffMapDir:
                    fitSlc(slc, False, t2, b1, ff)
                else:
                    fitSlcFast(slc, False, t2, b1, ff)

print("Elapsed time", time.time() - t)

save3dDicom(t2*10, infoOut, os.path.join(baseDir, 't2' + outSuffix), 97)
save3dDicom(b1*100, infoOut, os.path.join(baseDir, 'b1' + outSuffix), 98)
save3dDicom(ff*100, infoOut, os.path.join(baseDir, 'ff' + outSuffix), 99)
    
