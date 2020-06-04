# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 13:59:52 2016

@author: francesco
"""

import math
import numpy as np

# new implementation, working on a generic number of axes
def padorcut(arrayin, newSize, axis = None):
    nDims = arrayin.ndim
    
    # extend dimensions
    while nDims < len(newSize):
        arrayin = np.expand_dims(arrayin, nDims)
        nDims = arrayin.ndim
        
    if type(axis) is int:
        # check if newsz is iterable, otherwise assume it's a number
        try:
            newSz = newSize[axis]
        except:
            newSz = int(newSize)
        oldSz = arrayin.shape[axis]
        if oldSz < newSz:
            padBefore = int(math.floor(float(newSz - oldSz)/2))
            padAfter = int(math.ceil(float(newSz - oldSz)/2))
            padding = []
            for i in range(nDims):
                if i == axis:
                    padding.append( (padBefore, padAfter) )
                else:
                    padding.append( (0,0) )
            return np.pad(arrayin, padding, 'constant')
        elif oldSz > newSz:
            cutBefore = int(math.floor(float(oldSz - newSz)/2))
            cutAfter = int(math.ceil(float(oldSz - newSz)/2))
            slc = [slice(None)]*nDims
            slc[axis] = slice(cutBefore, -cutAfter)
            return arrayin[tuple(slc)]
        else:
            return arrayin
    else:
        for ax in range(nDims):
            arrayin = padorcut(arrayin, newSize, ax)
        return arrayin
        
                
            

def padorcut_old(arrayin, newsize):
    # function arrayout = padorcut(arrayin, newsize)
    # Symmetrically pads or cuts an array (2D!!) so that its final size is newsize
    
    oldsize = arrayin.shape

    
    if oldsize[0] > newsize[0]:
        
        # we have to cut
        
        cutbefore = int(math.floor( float(oldsize[0] - newsize[0] ) / 2 ))
        cutafter = int(math.ceil( float( oldsize[0] - newsize[0] ) / 2 ))
        
        arraytemp = arrayin[cutbefore:(oldsize[0]-cutafter), :]
        
    elif oldsize[0] < newsize[0]:
        
        # we have to pad
        padbefore = int(math.floor( float( newsize[0] - oldsize[0] ) / 2))
        padafter = int(math.ceil( float( newsize[0] - oldsize[0] ) / 2))
        
        arraytemp=np.pad(arrayin, ( (padbefore, padafter), (0,0) ), 'constant')   
    else: # oldsize[0] == newsize[0]
        arraytemp=arrayin;

    
    # second direction
    
    if oldsize[1] > newsize[1]:
        
        # we have to cut
        
        cutbefore = int(math.floor( float( oldsize[1] - newsize[1] ) / 2 ))
        cutafter = int(math.ceil( float( oldsize[1] - newsize[1] ) / 2 ))

        
        arrayout = arraytemp[:, cutbefore:(oldsize[1]-cutafter)];
        
        
    elif oldsize[1] < newsize[1]:
        
        # we have to pad
        
        padbefore = int(math.floor( float( newsize[1] - oldsize[1] ) / 2))
        padafter = int(math.ceil( float( newsize[1] - oldsize[1] ) / 2))
        
        
        arrayout=np.pad(arraytemp, ( (0,0), (padbefore, padafter) ), 'constant')
        
    else: # oldsize[1] == newsize[1]       
        arrayout=arraytemp;
        
    return arrayout

def translate(arrayin, translation):
    padBeforeX = translation[0] if translation[0] > 0 else 0
    padAfterX = -translation[0] if translation[0] < 0 else 0
    padBeforeY = translation[1] if translation[1] > 0 else 0
    padAfterY = -translation[1] if translation[1] < 0 else 0
    
    cutBeforeX = padAfterX
    cutAfterX = padBeforeX
    cutBeforeY = padAfterY
    cutAfterY = padBeforeY
    
    nx, ny = arrayin.shape    
    
    cutArray = arrayin[cutBeforeX:(nx-cutAfterX), cutBeforeY:(ny-cutAfterY)]
    arrayout = arrayout=np.pad(cutArray, ( (padBeforeX,padAfterX), (padBeforeY, padAfterY) ), 'constant')
    return arrayout
    
    