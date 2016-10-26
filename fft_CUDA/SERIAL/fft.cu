#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "fft.h"


unsigned int bit_reverse_CPU(unsigned int n, unsigned int bits){
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

void fft_by_row_CPU(float *real,float *imag, int rows, int cols){
  float w_real, w_imaginery;
  int row, col;
  int stages = round(log2((float)rows));
  int index_reversed;
  int reversed;
  float swap, current_cos, current_sin;
  int stage;
  int index;
  unsigned  int stride = 1;
  int dft_size = 2;
  int i, j;
  int shift;

  for (row = 0; row < rows; row++){
    /* reverse bits */
    for(col = 0;col < cols ; col++){
      index = row * cols + col;
      index_reversed = bit_reverse_CPU(col, stages);
      if(index_reversed > col){
        reversed = (row * cols) + index_reversed;
        swap = real[index];
        real[index] = real[reversed];
        real[reversed] = swap;
        swap = imag[index];
        imag[index] = imag[reversed];
        imag[reversed] = swap;
      }
    }
    shift = (row * cols);
    for(stage = 0, stride = 1, dft_size = 2; stage < stages; stage++, stride <<= 1, dft_size <<= 1){ 
      for(i = 0; i < stride; i++){
        current_cos = cos(2.0 *(M_PI * i) / dft_size);
        current_sin = sin(2.0 *(M_PI * i) / dft_size);
        for(j = i; j < cols; j+= dft_size){
          w_real = (current_cos * real[j + stride + shift]) + (current_sin * imag[j + stride + shift]);
          w_imaginery = (current_cos * imag[j + stride + shift]) - (current_sin * real[j + stride + shift]);
          real[j + stride + shift] = real[j + shift] - w_real;
          imag[j + stride + shift] = imag[j + shift] - w_imaginery;
          real[j + shift] += w_real;
          imag[j + shift] += w_imaginery;
        }
      }
    }
  }
}

void fft_by_col_CPU(float *real,float *imag, int rows, int cols){
  float w_real, w_imaginery;
  int row, col;
  int stages = round(log2((float)cols));
  int index_reversed;
  int reversed;
  float swap, current_cos, current_sin;
  int stage;
  int index;
  unsigned  int stride = 1;
  int dft_size = 2;
  int i, j;
  int shift;

  for (col = 0; col < cols; col++){
    /* reverse bits */
    for(row = 0;row <  rows; row++){
      index = row * cols + col;
      index_reversed = bit_reverse_CPU(row, stages);
      if(index_reversed > row){
        reversed = (index_reversed * cols) + col;
        swap = real[index];
        real[index] = real[reversed];
        real[reversed] = swap;
        swap = imag[index];
        imag[index] = imag[reversed];
        imag[reversed] = swap;
      }
    }

    for(stage = 0, stride = 1, dft_size = 2; stage < stages; stage++, stride <<= 1, dft_size <<= 1){
      for(i = 0; i < stride; i++){
        current_cos = cos(2.0 *(M_PI * i) / dft_size);
        current_sin = sin(2.0 *(M_PI * i) / dft_size);
        for(j = i; j < rows; j+= dft_size){
          shift = (j + stride) * cols;
          w_real = (current_cos * real[shift + col]) + (current_sin * imag[shift + col]);
          w_imaginery = (current_cos * imag[shift + col]) - (current_sin * real[shift + col]);
          real[shift + col] = real[j * cols + col] - w_real;
          imag[shift + col] = imag[j * cols + col] - w_imaginery;
          real[j * cols + col] += w_real;
          imag[j * cols + col] += w_imaginery;
        }
      }
    }
  }
}
