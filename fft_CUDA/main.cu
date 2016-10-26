#include <stdio.h>
#include <stdlib.h>

#include "GPU/fft.h"
#include "SERIAL/fft.h"

float GPU(const char *filename, const char *output){
 // MEMORY ALLOCATION
  t_image *image = read_ppm_image(filename);
  t_gray_image *gray_image = image_2_gray_image(image);
  t_complex_image *complex_image = gray_image_2_complex_image(gray_image);
  int bits;
  float *DEV_REAL, *DEV_IMAG;
  int *DEV_NBITS, *DEV_W;

  // SHIFT FREQUENCY 
  shift_frequency_domain(complex_image);
  
  // NUMBER OF BITS TO REPRESENT A SINGLE ROW
  bits = (int)log2((float)complex_image->y);

  // CUDA EVENTS
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // START TIME GPU
  cudaEventRecord(start);

  // MEMORY ALLOCATION ON GPU
  cudaMalloc((void**)&DEV_REAL, sizeof(float)*(complex_image->x * complex_image->y));
  cudaMalloc((void**)&DEV_IMAG, sizeof(float)*(complex_image->x * complex_image->y));
  cudaMalloc((void**)&DEV_NBITS, sizeof(int));
  cudaMalloc((void**)&DEV_W, sizeof(int));

  // COPY DATA TO GPU
  cudaMemcpy(DEV_REAL, complex_image->real, sizeof(float)*(complex_image->x * complex_image->y), cudaMemcpyHostToDevice);
  cudaMemcpy(DEV_IMAG, complex_image->imag, sizeof(float)*(complex_image->x * complex_image->y), cudaMemcpyHostToDevice);
  cudaMemcpy(DEV_NBITS, &bits, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(DEV_W, &complex_image->y, sizeof(int), cudaMemcpyHostToDevice);

  // BLOCKS
  dim3 blocks_row(complex_image->x, 1, 1);

  // THREADS PER BLOCK
  dim3 threads_row(complex_image->y, 1, 1);

  // KERNEL TO COMPUTE ALL ROWS ON GPU
  fft_by_row_GPU<<<blocks_row, threads_row>>>(DEV_REAL,DEV_IMAG, DEV_NBITS);

  // NUMBER OF BITS TO REPRESENT A SINGLE COLUMN
  bits = (int)log2((float)complex_image->x);

  // BLOCKS
  dim3 blocks_col(1,complex_image->y, 1);

  // THREADS PER BLOCK
  dim3 threads_col(1, complex_image->x, 1);

  // COPY DATA TO GPU
  cudaMemcpy(DEV_NBITS, &bits, sizeof(int), cudaMemcpyHostToDevice);

  // KERNEL TO COMPUTE ALL COLUMNS ON GPU
  fft_by_col_GPU<<<blocks_col, threads_col>>>(DEV_REAL,DEV_IMAG, DEV_NBITS, DEV_W);

  // COPY RESULT TO CPU
  cudaMemcpy(complex_image->real, DEV_REAL,sizeof(float)*(complex_image->x * complex_image->y), cudaMemcpyDeviceToHost);
  cudaMemcpy(complex_image->imag, DEV_IMAG,sizeof(float)*(complex_image->x * complex_image->y), cudaMemcpyDeviceToHost);

  // END TIME GPU
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // COMPUTE TOTAL TIME GPU
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // OUTPUT DATA
  t_gray_image *spectre =  image_spectre(complex_image);
  write_ppm_gray_image(output, spectre);

  // FREE MEMORY
  free_image(image);
  free_gray_image(gray_image);
  free_complex_image(complex_image);
  free_gray_image(spectre);

  // FREE MEMORY ON GPU
  cudaFree(DEV_REAL);
  cudaFree(DEV_IMAG);
  cudaFree(DEV_NBITS);
  cudaFree(DEV_W);

  return milliseconds;
}

float CPU(const char *filename, const char *output){
 // MEMORY ALLOCATION
  t_image *image = read_ppm_image(filename);
  t_gray_image *gray_image = image_2_gray_image(image);
  t_complex_image *complex_image = gray_image_2_complex_image(gray_image);

  // SHIFT FREQUENCY 
  shift_frequency_domain(complex_image);

  // CUDA EVENTS
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // START TIME SERIAL
  cudaEventRecord(start);

  // SERIAL FFT, ALL ROWS AND COLUMNS
  fft_by_row_CPU(complex_image->real, complex_image->imag, complex_image->x, complex_image->y);
  fft_by_col_CPU(complex_image->real, complex_image->imag, complex_image->x, complex_image->y);
  
  // END TIME GPU
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  // COMPUTE TOTAL SERIAL
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  // OUTPUT DATA
  t_gray_image *spectre =  image_spectre(complex_image);
  write_ppm_gray_image(output, spectre);
  
  // FREE MEMORY
  free_image(image);
  free_gray_image(gray_image);
  free_complex_image(complex_image);
  free_gray_image(spectre);

  return milliseconds; 
}


void execute(){
  float cpu, gpu;
  gpu = GPU("INPUT/input01.ppm", "OUTPUT/gpu-input01.ppm");
  cpu = CPU("INPUT/input01.ppm", "OUTPUT/cpu-input01.ppm");
  printf("\nINPUT/input01.ppm-----------------\n\nSpeedup: %.3f\n", (cpu / gpu));
  gpu = GPU("INPUT/input02.ppm", "OUTPUT/gpu-input02.ppm");
  cpu = CPU("INPUT/input02.ppm", "OUTPUT/cpu-input02.ppm");
  printf("\nINPUT/input02.ppm-----------------\n\nSpeedup: %.3f\n", (cpu / gpu));
  gpu = GPU("INPUT/input03.ppm", "OUTPUT/gpu-input03.ppm");
  cpu = CPU("INPUT/input03.ppm", "OUTPUT/cpu-input03.ppm");
  printf("\nINPUT/input03.ppm-----------------\n\nSpeedup: %.3f\n", (cpu / gpu));
}

int main(int argc, char *argv[]) {
  execute();
  return 0;
}
