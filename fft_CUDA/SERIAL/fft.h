#ifndef FFT_SERIAL

#define FFT_SERIAL
#define PI 3.14159265

unsigned int bit_reverse_CPU(unsigned int n, unsigned int bits);
void fft_by_row_CPU(float *real,float *imag, int rows, int cols);
void fft_by_col_CPU(float *real,float *imag, int rows, int cols);

#endif