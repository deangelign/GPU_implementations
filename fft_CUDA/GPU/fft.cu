#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "fft.h"



__device__ unsigned int bit_reverse_GPU(unsigned int n, unsigned int bits){
  unsigned int nrev, N;
  unsigned int count;
  N = 1 << bits;
  count = bits - 1;
  nrev = n;
  for(n >>= 1; n; n >>= 1)
  {
    nrev <<= 1;
    nrev |= n & 1;
    count--;
  }

  nrev <<= count;
  nrev &= N - 1;

  return nrev;
}

__global__ void fft_by_row_GPU(float *real,float *imag, int *bits){
  __shared__ float real_shared[1024];
  __shared__ float imag_shared[1024];

  float w_real, w_imag;
  t_complex x, y;
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  int stages = (*bits);
  int stage;
  int factor;

  /*reverse bit*/
  unsigned  int reversed = bit_reverse_GPU(threadIdx.x, (*bits));
  real_shared[reversed] = real[i];
  imag_shared[reversed] = 0;
  unsigned  int stride = 1;
  int dft_size = 2;
  int theta;

  __syncthreads();

  for(stage = 0; stage < stages; stage++){

    factor = (((threadIdx.x % (2 * stride)) / stride) % 2 == 0) ? 1 : -1;
    theta = threadIdx.x % stride;
    x.real = real_shared[threadIdx.x];
    x.imag = imag_shared[threadIdx.x];
    if(factor == 1){
      y.real = real_shared[threadIdx.x + stride];
      y.imag = imag_shared[threadIdx.x + stride];
    }else{
      y.real = real_shared[threadIdx.x - stride];
      y.imag = imag_shared[threadIdx.x - stride];	
    }

    __syncthreads();

    if(factor == 1){
      w_real = y.real * cos(2 * PI * theta / dft_size) + (sin(2 * PI * theta / dft_size) * y.imag);
      w_imag = y.imag * cos(2 * PI * theta / dft_size) - (sin(2 * PI * theta / dft_size) * y.real);
      real_shared[threadIdx.x] = x.real + w_real;
      imag_shared[threadIdx.x] = x.imag + w_imag;

    }else{
      w_real = x.real * cos(2 * PI * theta / dft_size) + (sin(2 * PI * theta / dft_size) * x.imag);
      w_imag = x.imag * cos(2 * PI * theta / dft_size) - (sin(2 * PI * theta / dft_size) * x.real);
      real_shared[threadIdx.x] = y.real - w_real;
      imag_shared[threadIdx.x] = y.imag - w_imag;
    }

    __syncthreads();
    stride <<= 1;
    dft_size <<= 1;
  }

  real[i] = real_shared[threadIdx.x];
  imag[i] = imag_shared[threadIdx.x];
}




__global__ void fft_by_col_GPU(float *real,float *imag, int *bits, int *w){
  __shared__ float real_shared[1024];
  __shared__ float imag_shared[1024];

  float w_real, w_imag;
  t_complex x, y;
  int i = threadIdx.y * *w + blockIdx.y;
  int stages = (*bits);
  int stage;
  int factor;

  /*reverse bit*/
  unsigned  int reversed = bit_reverse_GPU(threadIdx.y, (*bits));
  real_shared[reversed] = real[i];
  imag_shared[reversed] = imag[i];
  unsigned  int stride = 1;
  int dft_size = 2;
  int theta;

  __syncthreads();

  for(stage=0; stage < stages; stage++){

    factor = (((threadIdx.y % (2 * stride)) / stride) % 2 == 0) ? 1 : -1;
    theta = threadIdx.y % stride;
    x.real = real_shared[threadIdx.y];
    x.imag = imag_shared[threadIdx.y];
    if(factor == 1){
      y.real = real_shared[threadIdx.y + stride];
      y.imag = imag_shared[threadIdx.y + stride];
    }else{
      y.real = real_shared[threadIdx.y - stride];
      y.imag = imag_shared[threadIdx.y - stride];	
    }

    __syncthreads();

    if(factor == 1){
      w_real = y.real * cos(2 * PI * theta / dft_size) + (sin(2 * PI * theta / dft_size) * y.imag);
      w_imag = y.imag * cos(2 * PI * theta / dft_size) - (sin(2 * PI * theta / dft_size) * y.real);
      real_shared[threadIdx.y] = x.real + w_real;
      imag_shared[threadIdx.y] = x.imag + w_imag;

    }else{
      w_real = x.real * cos(2 * PI * theta / dft_size) + (sin(2 * PI * theta / dft_size) * x.imag);
      w_imag = x.imag * cos(2 * PI * theta / dft_size) - (sin(2 * PI * theta / dft_size) * x.real);
      real_shared[threadIdx.y] = y.real - w_real;
      imag_shared[threadIdx.y] = y.imag - w_imag;
    }

    __syncthreads();
    stride <<= 1;
    dft_size <<= 1;
  }

  real[i] = real_shared[threadIdx.y];
  imag[i] = imag_shared[threadIdx.y];
}
