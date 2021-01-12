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

from epg_sim import cpmg
import numpy as np
import numpy.fft as fft
import bisect
import os.path
import numpy.linalg as linalg
import scipy.optimize as optimize
import matplotlib.pyplot as plt

from SlrTransform import slr 
from scipy.signal import freqz

DATADIR = 'data'

def calcSliceprof_fft(nomFA, tbw): 
    t = np.linspace(-tbw/2, tbw/2, 100)
    pulse = np.multiply(np.hanning(len(t)), np.sinc(t))
    total = np.sum(pulse)
    pulse = np.radians(pulse / total * nomFA)
    
    # alternative a bit less precise but errors ~1deg
    h = fft.fft(fft.fftshift(np.pad(pulse, (1000,1000))))
    sliceprof = np.real(h[0:199]) # real is a bit risky because it only works for symmetric pulses, but it accounts for opposite phase
    sliceprof = sliceprof/np.max(sliceprof)*nomFA
    return sliceprof

def calcSliceprof_slr(nomFA, tbw): 
    t = np.linspace(-tbw/2, tbw/2, 100)
    pulse = np.multiply(np.hanning(len(t)), np.sinc(t))
    total = np.sum(pulse)
    pulse = np.radians(pulse / total * nomFA)
    
    a,b = slr(pulse)
    w,h = freqz(b,a,1024)
    sliceprof = np.abs(h[0:199]) # real is a bit risky because it only works for symmetric pulses

    # alternative a bit less precise but errors ~1deg
    #h = fft.fft(fft.fftshift(np.pad(pulse, (1000,1000))))
    #sliceprof = np.real(h[0:199]) # real is a bit risky because it only works for symmetric pulses, but it accounts for opposite phase
    sliceprof = sliceprof/np.max(sliceprof)*nomFA
    return sliceprof

# binning of the slice profile
def reduceSliceProf(sliceprof, bins, lastVal = None):
    if not lastVal:
        lastVal = np.argwhere( np.abs(sliceprof > 0.9) ).max()
    sliceprof = sliceprof[:lastVal]
    
    binsize = np.ceil(len(sliceprof) / bins);
    sliceprof_out = np.zeros((bins,1));
    
    for i in range(bins):
        startIndex = int(i*binsize)
        endIndex = int((i+1)*binsize)
        if endIndex > len(sliceprof):
            endIndex = len(sliceprof)
        if startIndex >= endIndex:
            sliceprof_out[i] = 0
        else:
            sliceprof_out[i] = sliceprof[startIndex:endIndex].mean()
    
    return sliceprof_out.squeeze(), lastVal
        


class FatFractionLookup:
    
    T1w = 1400
    T1f = 365
    TBW = 2.0
#    NT2s = 200 # number of calculated T2 points
#    NB1s = 50 # number of calculated B1 points
    NT2s = 60 # number of calculated T2 points
    NB1s = 20 # number of calculated B1 points
    MagPreparePulse = False
    NFF = 101
    
    def __init__(self, T2Limits, B1Limits, FatT2, NEchoes, EchoSpacing):
        self.fatT2 = FatT2
        self.NEchoes = NEchoes
        self.EchoSpacing = EchoSpacing
        self.T2Limits = T2Limits
        self.B1Limits = B1Limits

        self.rfParameters = None
        self.setPulses() # use default values for setpulses
        
        self.T2Points = np.linspace(T2Limits[0], T2Limits[1], self.NT2s)
        self.B1Points = np.linspace(B1Limits[0], B1Limits[1], self.NB1s)
        
        self.signalsReady = False
        
    
    def generateSignals(self):
        filename = "_t2l{:.3f}_{:.3f}_b1l{:.3f}_{:.3f}_ft2{:.3f}_etl{:d}_s{:.3f}_ed{:d}_ebw{:.3f}_rd{:d}_rbw{:.3f}_rwf{:.3f}.npy".format(*self.T2Limits, *self.B1Limits, self.fatT2, self.NEchoes, self.EchoSpacing, *self.rfParameters)
        print(filename)
        
        try:
            self.waterSignals = np.load(os.path.join(DATADIR, "water" + filename))
            self.fatSignals = np.load(os.path.join(DATADIR, "fat" + filename))
        except:
            print("Generating water signals")
            self.waterSignals = np.zeros((self.NT2s, self.NB1s, self.NEchoes))
            for t2Index in range(self.NT2s):
                for b1Index in range(self.NB1s):
                    self.waterSignals[t2Index, b1Index, :] = self._signalCalc(self.T1w, self.T2Points[t2Index], self.B1Points[b1Index])
            
            print("Generating fat signals")
            self.fatSignals = np.zeros((self.NB1s, self.NEchoes))
            for b1Index in range(self.NB1s):
                self.fatSignals[b1Index, :] = self._signalCalc(self.T1f, self.fatT2, self.B1Points[b1Index])
            
            print("Signals generated")
            try:
                np.save(os.path.join(DATADIR, "water" + filename), self.waterSignals)
                np.save(os.path.join(DATADIR, "fat" + filename), self.fatSignals)
            except:
                print("Warning: signals could not be saved. Next time they will be generated again. Make sure that the folder '%s' exists" % (DATADIR))
            
            self.signalsReady = True
    
    def setPulses(self, excDeg = 90, excBW = TBW, refDeg = 180, refBW = TBW, refWidthFactor = 0.2):
        self.rfParameters = [excDeg, excBW, refDeg, refBW, refWidthFactor]
        
        EXC_SAMPLES = 20
        refSamples = round(EXC_SAMPLES * (1+refWidthFactor))
        
        self.sliceProf90, lastVal = reduceSliceProf(calcSliceprof_slr(excDeg, excBW), EXC_SAMPLES)
        #self.sliceProf90, lastVal = reduceSliceProf(calcSliceprof_fft(excDeg, excBW), EXC_SAMPLES)
        self.sliceProf90 = np.pad(self.sliceProf90, (0,refSamples - EXC_SAMPLES) )
        self.sliceProf180, _ = reduceSliceProf(calcSliceprof_slr(refDeg/2, refBW)*2,refSamples, lastVal) # SLR transform doesn't work for 180°!
        #self.sliceProf180, _ = reduceSliceProf(calcSliceprof_fft(180, refBW),refSamples, lastVal) # SLR transform doesn't work for 180°!
        #sliceProf180_old, _ = reduceSliceProf(calcSliceprof_fft(180, refBW),refSamples) # SLR transform doesn't work for 180°!
        self.sliceProf90[np.isnan(self.sliceProf90)] = 0.0
        self.sliceProf180[np.isnan(self.sliceProf180)] = 0.0
        #print("90",self.sliceProf90)
        #print("180 old", sliceProf180_old)
        #print("180 new", self.sliceProf180)
        #invalidate signals
        self.waterSignals = None
        self.fatSignals = None
        self.signalsReady = False
        
    
    def _signalCalc(self, T1, T2, B1Factor):
        signal = np.zeros((self.NEchoes), dtype=np.complex)
        for curFAIndex in range(0, len(self.sliceProf90)): #Slice profile
            signal += cpmg(self.NEchoes, self.sliceProf90[curFAIndex]*B1Factor, self.sliceProf180[curFAIndex]*B1Factor, self.EchoSpacing, T1, T2, self.MagPreparePulse)
        #signal /= len(self.sliceProf90)
        signal /= signal.max()
        

        return np.abs(signal).astype(np.float)
               
    # returns T2 and B1
    def cpmgFit(self, yVector, T1):
        def objFun(params):
            a = params[0]
            t2 = params[1]
            b1 = params[2]
            
            signal = a * self._signalCalc(T1, t2, b1)
            # plt.cla()
            # plt.plot(signal)
            # plt.plot(yVector)
            # plt.pause(0.0001)
            # plt.show()
            return linalg.norm(signal  - yVector)
        
        optParam = optimize.minimize(objFun, (yVector.max(), 40, 1), bounds = ((0,1000000), (10, 300), (0.8, 1.2)) ).x
        return optParam[1], optParam[2]
        
       
    # returns a signal for a given water T1, B1 factor and fat fraction
    # deprecated
    def getSignal(self, T2, B1, fatFraction):
        if not self.signalsReady: self.generateSignals()
        # interpolation of the signals
        t2Weights = np.zeros((2,1))
        lowerT2Index = bisect.bisect_right(self.T2Points, T2)
        higherT2Index = lowerT2Index+1
        if lowerT2Index >= len(self.T2Points):
            lowerT2Index = len(self.T2Points) - 1
#            higherT2Index = lowerT2Index # if this condition is satisfied, the one below as well, so this is redundant
#            t2Weights[0] = 1.0
#            t2Weights[1] = 0.0
        if higherT2Index >= len(self.T2Points) or lowerT2Index == 0: # out of boundaries
            higherT2Index = lowerT2Index
            t2Weights[0] = 1.0
            t2Weights[1] = 0.0
        else:
            t2Weights[0] = self.T2Points[higherT2Index] - T2
            t2Weights[1] = T2 - self.T2Points[lowerT2Index]
            t2Weights /= t2Weights.sum()
            
        b1Weights = np.zeros((2,1))
        lowerB1Index = bisect.bisect_right(self.B1Points, B1)
        higherB1Index = lowerB1Index+1
        if lowerB1Index >= len(self.B1Points):
            lowerB1Index = len(self.B1Points) - 1
#            higherB1Index = lowerB1Index # if this condition is satisfied, the one below as well, so this is redundant
#            B1Weights[0] = 1.0
#            B1Weights[1] = 0.0
        if higherB1Index >= len(self.B1Points) or lowerB1Index == 0: # out of boundaries
            higherB1Index = lowerB1Index
            b1Weights[0] = 1.0
            b1Weights[1] = 0.0
        else:
            b1Weights[0] = self.B1Points[higherB1Index] - B1
            b1Weights[1] = B1 - self.B1Points[lowerB1Index]
            b1Weights /= b1Weights.sum()
            
        waterSignal = self.waterSignals[lowerT2Index,lowerB1Index,:]*t2Weights[0]*b1Weights[0] + self.waterSignals[higherT2Index,lowerB1Index,:]*t2Weights[1]*b1Weights[0] + self.waterSignals[lowerT2Index,higherB1Index,:]*t2Weights[0]*b1Weights[1] + self.waterSignals[higherT2Index,higherB1Index,:]*t2Weights[1]*b1Weights[1]
        fatSignal = self.fatSignals[lowerB1Index,:]*b1Weights[0] + self.fatSignals[higherB1Index,:]*b1Weights[1]
        
        signal = waterSignal * (1-fatFraction) + fatSignal * fatFraction
        return signal
    
    def getAllSignals(self):
        if not self.signalsReady: self.generateSignals()
        signalsOut = np.zeros( (self.NT2s * self.NB1s * self.NFF, self.NEchoes ) )
        parameterCombinations = []
        ffVector = np.linspace(0,1,self.NFF)
        curSignalIndex = 0
        for ffInd in range(len(ffVector)):
            for t2Ind in range(len(self.T2Points)):
                for b1Ind in range(len(self.B1Points)):
                    sig = (1-ffVector[ffInd]) * self.waterSignals[t2Ind, b1Ind,:] + ffVector[ffInd] * self.fatSignals[b1Ind,:]
                    signalsOut[curSignalIndex,:] = sig/sig[0]
                    #signalsOut[curSignalIndex,:] = self.getSignal(self.T2Points[t2Ind], self.B1Points[b1Ind], ffVector[ffInd])
                    parameterCombinations.append( (self.T2Points[t2Ind], self.B1Points[b1Ind], ffVector[ffInd]) )
                    curSignalIndex += 1
                    
        
        return np.array(parameterCombinations), signalsOut
            
        





