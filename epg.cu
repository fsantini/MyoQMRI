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
    
    Copyright 2019 Francesco Santini <francesco.santini@unibas.ch>

*/

#ifndef PYCUDA_COMPILE

    #define NECHOES 8
    #define ECHOSPACING 8.5f
    #define T1F 365.0f
    #define T1W 1400.0f
    #define MAGPREP 0

#endif
    
#include <iostream>
#include <math.h>
#include <thrust/fill.h>
#include <thrust/copy.h>
#include <thrust/complex.h>
#include <cuComplex.h>

// row-major indexing
#define IDX2C(i,j,nRows) (((j)*(nRows))+(i))
#define IDX2C3(i,j) IDX2C(i,j,3)

using complexType = thrust::complex<float>;
using cuType = cuFloatComplex;

template <typename matrixType, int N>
__device__ void dephase(matrixType *stateMatrix)
{
    for (int i=0; i<N-1; i++)
    {
        stateMatrix[IDX2C3(0,N-i-1)] = stateMatrix[IDX2C3(0,N-i-2)];
        stateMatrix[IDX2C3(1,i)] = stateMatrix[IDX2C3(1,i+1)];
    }
    stateMatrix[IDX2C3(0,0)] = thrust::conj(stateMatrix[IDX2C3(1,0)]);
    stateMatrix[IDX2C3(1, N-1)] = 0.0;
}

template <typename matrixType, int N>
__device__ void relax(matrixType *stateMatrix, const matrixType *relaxMatrix)
{
    matrixType z0 = stateMatrix[IDX2C3(2,0)];
    for (int i=0; i<N*3; i++)
    {
        stateMatrix[i] *= relaxMatrix[i];
    }
    stateMatrix[IDX2C3(2,0)] += 1 - z0;
}

template <typename matrixType, int N>
__device__ void rfMult(const matrixType *tMatrix, const matrixType *stateMatrix, matrixType *outMatrix)
{
    for (auto c=0; c<N; c++)
    {
        for (auto r=0; r<3; r++)
        {
            matrixType sum = 0;
            for (auto rc=0; rc<3; rc++)
            {
                sum += tMatrix[IDX2C3(r,rc)]*stateMatrix[IDX2C3(rc,c)];
            }
            outMatrix[IDX2C3(r,c)] = sum;
        }
    }
}

template <int Nechoes>
__device__ void cpmg(float exc_alpha, float ref_alpha, float T1, float T2, thrust::complex<float> *outVector)
{
    
    const int N = Nechoes;
    const int Nt2 = 2*N;
    const int Nt2p1 = Nt2+1;
    const complexType jp(0.0,1.0);
    const complexType jm(0.0,-1.0);
    
    float alpha_in = M_PI*ref_alpha/180;
    float exc_alpha_in = M_PI*exc_alpha/180;
    
    complexType fa[2];
    
    fa[1] = alpha_in;
    fa[0] = alpha_in;
    
    if (N>1 && MAGPREP)
    {
        fa[0] = (M_PI + fa[1])/2;
    }

    float E1 = exp(-ECHOSPACING/T1/2.0);
    float E2 = exp(-ECHOSPACING/T2/2.0);
    
    complexType RelaxMatrix[3*Nt2p1];
    for (int i=0; i<Nt2p1; i++)
    {
        RelaxMatrix[IDX2C3(0,i)] = complexType(E2,0.0f);
        RelaxMatrix[IDX2C3(1,i)] = complexType(E2,0.0f);
        RelaxMatrix[IDX2C3(2,i)] = complexType(E1,0.0f);
    }
    
    complexType Omega_preRF[3*Nt2p1];
    complexType Omega_postRF[3*Nt2p1];
    
    for (auto i=0; i<3*Nt2p1; i++)
    {
        Omega_postRF[i] = complexType(0.0f, 0.0f);
        Omega_preRF[i] = complexType(0.0f, 0.0f);
    }
    
    Omega_postRF[IDX2C3(0,0)] = sin(exc_alpha_in);
    Omega_postRF[IDX2C3(1,0)] = sin(exc_alpha_in);
    Omega_postRF[IDX2C3(2,0)] = cos(exc_alpha_in);
    
    complexType tMatrix0[3*3];
    tMatrix0[IDX2C3(0,0)] =      pow(cos(fa[0]/2.0),2);
    tMatrix0[IDX2C3(0,1)] =      pow(sin(fa[0]/2.0),2);
    tMatrix0[IDX2C3(0,2)] = jm * sin(fa[0]);
    
    tMatrix0[IDX2C3(1,0)] =      pow(sin(fa[0]/2.0),2);
    tMatrix0[IDX2C3(1,1)] =      pow(cos(fa[0]/2.0),2);
    tMatrix0[IDX2C3(1,2)] = jp * sin(fa[0]);
    
    tMatrix0[IDX2C3(2,0)] = 0.5*jm* sin(fa[0]);
    tMatrix0[IDX2C3(2,1)] = 0.5*jp* sin(fa[0]);
    tMatrix0[IDX2C3(2,2)] =         cos(fa[0]);
    
    complexType tMatrix1[3*3];
    tMatrix1[IDX2C3(0,0)] =      pow(cos(fa[1]/2.0),2);
    tMatrix1[IDX2C3(0,1)] =      pow(sin(fa[1]/2.0),2);
    tMatrix1[IDX2C3(0,2)] = jm * sin(fa[1]);
    
    tMatrix1[IDX2C3(1,0)] =      pow(sin(fa[1]/2.0),2);
    tMatrix1[IDX2C3(1,1)] =      pow(cos(fa[1]/2.0),2);
    tMatrix1[IDX2C3(1,2)] = jp * sin(fa[1]);
    
    tMatrix1[IDX2C3(2,0)] = 0.5*jm* sin(fa[1]);
    tMatrix1[IDX2C3(2,1)] = 0.5*jp* sin(fa[1]);
    tMatrix1[IDX2C3(2,2)] =         cos(fa[1]);
    
    // first relaxation
    //printf("Omega_postRF[0,0]: %f\n", Omega_postRF[0].real());
    relax<complexType, Nt2p1>(Omega_postRF, RelaxMatrix);
    //printf("Omega_postRF[0,0]: %f\n", Omega_postRF[0].real());
    dephase<complexType, Nt2p1>(Omega_postRF);
    //printf("Omega_postRF[0,0]: %f\n", Omega_postRF[0].real());
    
    // first refocusing RF
    
    rfMult<complexType, Nt2p1>(tMatrix0, Omega_postRF, Omega_preRF);
    //printf("Omega_preRF[0,0]: %f\n", Omega_preRF[0].real());
    // relaxation/recovery post refocusing
    relax<complexType, Nt2p1>(Omega_preRF, RelaxMatrix);
    //printf("Omega_preRF[0,0]: %f\n", Omega_preRF[0].real());
    dephase<complexType, Nt2p1>(Omega_preRF);
    //printf("Omega_preRF[0,0]: %f\n", Omega_preRF[0].real());
    
    outVector[0] = thrust::conj(Omega_preRF[IDX2C3(1,0)]);
    //printf("outvector[0] %f\n", outVector[0].real());
    
    thrust::copy(thrust::device, Omega_preRF, Omega_preRF+(3*Nt2p1), Omega_postRF); // copy state to other matrix
    
    for (int pn=1; pn<N; pn++)
    {
        // first relaxation
        
        relax<complexType, Nt2p1>(Omega_postRF, RelaxMatrix);
        dephase<complexType, Nt2p1>(Omega_postRF);
        
        
        // first refocusing RF
        
        rfMult<complexType, Nt2p1>(tMatrix1, Omega_postRF, Omega_preRF);
        
        // relaxation/recovery post refocusing
        relax<complexType, Nt2p1>(Omega_preRF, RelaxMatrix);
        dephase<complexType, Nt2p1>(Omega_preRF);
        
        outVector[pn] = thrust::conj(Omega_preRF[IDX2C3(1,0)]);
        //printf("outvector[%d] %f\n", pn, outVector[pn].real());
        thrust::copy(thrust::device, Omega_preRF, Omega_preRF+(3*Nt2p1), Omega_postRF); // copy state to other matrix
        
    }
    
}

// parameters is nx3: ff, t2, b1
// signals_out is nxNECHOES
extern "C" void __global__ cpmg_sliceprof_B1_FF(unsigned int totalParameters, unsigned int nFlipanglesSP, float T2f, float *flipAnglesEx, float *flipAnglesRef, float *parameters, float *signals_out)
{
    // calculate the cpmg signals for many values of B1 and Fat fractions
    const int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= totalParameters) return;
    float wT2 = parameters[index*3+0];
    float b1 = parameters[index*3+1];
    float ff = parameters[index*3+2];

    //printf("wT2 %f, b1 %f, ff %f\n", wT2, b1, ff);
    
    for (int echo=0; echo<NECHOES; echo++)
    {
        signals_out[index*NECHOES+echo] = 0.0;
    }
    
    for (int nFa = 0; nFa<nFlipanglesSP; nFa++)
    {
        complexType fatSignal[NECHOES];
        complexType waterSignal[NECHOES];
        cpmg<NECHOES>(flipAnglesEx[nFa]*b1, flipAnglesRef[nFa]*b1, T1F, T2f, fatSignal);
        cpmg<NECHOES>(flipAnglesEx[nFa]*b1, flipAnglesRef[nFa]*b1, T1W, wT2, waterSignal);
        //printf("SignalsOut: ");
        for (int echo=0; echo<NECHOES; echo++)
        {
            signals_out[index*NECHOES+echo] += float( (fatSignal[echo]*ff).real() + (waterSignal[echo]*(1-ff)).real() )/nFlipanglesSP;
            //printf("%f ", signals_out[index*NECHOES+echo]);
        }
        //printf("\n");
    }
}

// Test

#ifndef PYCUDA_COMPILE

#define T2F 151.0f

#define NFF 1
#define NT2 6
#define NB1 2

#define minT2 20.0
#define maxT2 80.0

#define minB1 0.6
#define maxB1 1.4

#define minFF 0.0
#define maxFF 1.0

__global__ void createParams(float *params, float *spExc, float *spRef)
{
    spExc[0] = 45;
    spExc[1] = 90;
    spExc[2] = 45;
    
    spRef[0] = 90;
    spRef[1] = 180;
    spRef[2] = 90;
    
    int paramIndex = 0;
    // initialize x and y arrays on the host
    for (int nFF = 0; nFF<NFF; nFF++)
    {
        for (int nT2 = 0; nT2 < NT2; nT2++)
        {
            for (int nB1 = 0; nB1 < NB1; nB1++)
            {
                //printf("paramIndex %d\n", paramIndex);
                params[paramIndex++] = float(nT2)*(maxT2-minT2)/NT2 + minT2;
                params[paramIndex++] = float(nB1)*(maxB1-minB1)/NB1 + minB1;
                params[paramIndex++] = float(nFF)*(maxFF-minFF)/NFF + minFF;
            }
        }
    }
}

int main(void)
{
    

  unsigned int Nparams = NT2*NFF*NB1;
  
  float *params, *signals;

  float *spExc;
  float *spRef;
  
  // Allocate Unified Memory accessible from CPU or GPU
  cudaMalloc((void**)&params, 3*Nparams*sizeof(float));
  
  cudaMalloc((void**)&spExc, 3*sizeof(float));
  cudaMalloc((void**)&spRef, 3*sizeof(float));
 
  std::cout << "Creating param space" << std::endl << std::flush;
  
  createParams<<<1,1>>>(params, spExc, spRef);
  
  cudaDeviceSynchronize();
  
  float *h_params = (float*)malloc(Nparams*3*sizeof(float));
  cudaMemcpy(h_params, params, Nparams*3*sizeof(float), cudaMemcpyDeviceToHost);
  
  std::cout << h_params[0] << ", " << h_params[1] << ", " << h_params[2] << std::endl;
  
  cudaMalloc((void**)&signals, NECHOES*Nparams*sizeof(float));
  
  std::cout << "Creating signals" << std::endl << std::flush;
  // Run kernel on 1M elements on the GPU
  int blockSize = 256;
  int nBlocks = ceil( float(Nparams)/blockSize );
  //cpmg_sliceprof_B1_FF<<< nBlocks, blockSize >>>(Nparams,  3, T2F, spExc, spRef, params, signals);
  cpmg_sliceprof_B1_FF<<< 1, 1 >>>(Nparams,  3, T2F, spExc, spRef, params, signals);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  float *h_signals = (float*)malloc(Nparams*NECHOES*sizeof(float));
  cudaMemcpy(h_signals, signals, Nparams*NECHOES*sizeof(float), cudaMemcpyDeviceToHost);
  
  std::cout << "Signals created" << std::endl << std::flush;

  std::cout << "Example: ";
  
  for (int i=0; i<NECHOES; i++)
  {
      std::cout << h_signals[0*NECHOES + i] << ", ";
  }
  std::cout << std::endl;
  
  cudaFree(params);
  cudaFree(signals);
  cudaFree(spExc);
  cudaFree(spRef);
  
  return 0;
}

#endif
