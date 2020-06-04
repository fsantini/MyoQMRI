/*

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

*/


#include <iostream>
#include <math.h>

// cublas matrices are column-major: index = row*nCol + col. We want the maximum over the columns

extern "C" void __global__ findmax_ff(float *corrMatrix, int *ffMatrix, unsigned int nVoxels, int *ffparameters, unsigned int nParams, unsigned int *outIndexVector)
{
    // corrMatrix will be nVoxelsxnParams
    // ffMatrix is nVoxelsx1
    // ffparameters is nParamsx1 - integer value of fat fraction in %
    // outIndexVector is nVoxelsx1
    const int voxelIndex = blockIdx.x*blockDim.x + threadIdx.x; // this will be the row if the corrMatrix
    
    if (voxelIndex >= nVoxels) return;
    float maxCorrVal = 0;
    int maxIndex = 0;
    for (int p=0; p<nParams; p++)
    {
        if (ffparameters[p] == ffMatrix[voxelIndex])
        {
            float curCorrVal = corrMatrix[voxelIndex*nParams + p];
            if (curCorrVal > maxCorrVal)
            {
                maxCorrVal = curCorrVal;
                maxIndex = p;
            }
        }
    }
    
    outIndexVector[voxelIndex] = maxIndex;
}