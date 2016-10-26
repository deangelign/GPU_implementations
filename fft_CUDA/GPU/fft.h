#ifndef FFT_GPU

#include "../UTIL/util.h"

#define FFT_GPU

__device__ unsigned int bit_reverse_GPU(unsigned int n, unsigned int bits);
__global__ void fft_by_row_GPU(float *real,float *imag, int *bits);
__global__ void fft_by_col_GPU(float *real,float *imag, int *bits, int *w);

#endif
