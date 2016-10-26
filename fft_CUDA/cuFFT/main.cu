#include <stdio.h>
#include <stdlib.h>

#include <cufft.h>

#define BSZ  32

__global__ void real2complex(float *f, cufftComplex *fc, int N){
  int i = threadIdx.x + blockIdx.x*blockDim.x;
  int j = threadIdx.y + blockIdx.y*blockDim.y;
  int index = j*N+i;

  if (i<N && j<N){

    fc[index].x = f[index];
    fc[index].y = 0.0f;
  }
}

__global__ void complex2real(cufftComplex *fc, float *f, int N){
  int i = threadIdx.x + blockIdx.x*BSZ;
  int j = threadIdx.y + blockIdx.y*BSZ;
  int index = j*N+i;
  if (i<N && j<N){
    f[index] = fc[index].x/((float)N*(float)N);
  }
}

int main(int argc, char *argv[]) {

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  int N = 1024;
  float xmax = 1.0f;
  float xmin = 0.0f;
  float ymin = 0.0f;
  float h = (xmax-xmin)/((float)N);
  float s = 0.1f;
  float s2 = s*s;
  float milliseconds;

  float *x = new float[N*N];
  float *y = new float[N*N];
  float *u = new float[N*N];
  float *f = new float[N*N];
  float *u_a = new float[N*N];
  float *err = new float[N*N];

  float r2;
  for(int j=0; j<N; j++){
    for(int i=0; i<N; i++){
      x[N*j+i] = xmin + i*h;
      y[N*j+i] = ymin + j*h;

      r2 = (x[N*j+i]-0.5)*(x[N*j+i]-0.5) + (y[N*j+i]-0.5)*(y[N*j+i]-0.5);
      f[N*j+i] = (r2-2*s2)/(r2*s2)*exp(-r2/(2*s2));
      u_a[N*j+i] = exp(-r2/(2*s2));
    }
  }

  float *k = new float[N];
  for (int i=0; i<=N/2; i++){
    k[i] = i * 2*M_PI;
  }
  for (int i=N/2+1; i<N; i++){
    k[i] = (i-N)* 2*M_PI;
  }

  float *k_d, *f_d, *u_d;
  cudaMalloc ((void**)&k_d, sizeof(float)*N);
  cudaMalloc ((void**)&f_d, sizeof(float)*N*N);
  cudaMalloc ((void**)&u_d, sizeof(float)*N*N);

  cudaEventRecord(start);
  cudaMemcpy(k_d, k, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(f_d, f, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  cufftComplex *ft_d, *f_dc, *ft_d_k, *u_dc;
  cudaMalloc ((void**)&ft_d, sizeof(cufftComplex)*N*N);
  cudaMalloc ((void**)&ft_d_k, sizeof(cufftComplex)*N*N);
  cudaMalloc ((void**)&f_dc, sizeof(cufftComplex)*N*N);
  cudaMalloc ((void**)&u_dc, sizeof(cufftComplex)*N*N);

  dim3 dimGrid (int(  (N-0.5)/BSZ )+1, int(  (N-0.5)/BSZ )+1,1  );
  dim3 dimBlock (BSZ, BSZ,1);
  real2complex<<<dimGrid, dimBlock>>>(f_d, f_dc, N);

  cufftHandle plan;
  cufftPlan2d(&plan, N, N, CUFFT_C2C);

  cufftExecC2C(plan, f_dc, ft_d, CUFFT_FORWARD);
  complex2real<<<dimGrid, dimBlock>>>(u_dc, u_d, N);

  cudaMemcpy(u, u_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("fft+criacao_plano: %0.3fms\n",milliseconds);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);


   N = 256;
   xmax = 1.0f;
   xmin = 0.0f;
   ymin = 0.0f;
   h = (xmax-xmin)/((float)N);
   s = 0.1f;
   s2 = s*s;

   x = new float[N*N];
   y = new float[N*N];
   u = new float[N*N];
   f = new float[N*N];
   u_a = new float[N*N];
   err = new float[N*N];


  for(int j=0; j<N; j++){
    for(int i=0; i<N; i++){
      x[N*j+i] = xmin + i*h;
      y[N*j+i] = ymin + j*h;

      r2 = (x[N*j+i]-0.5)*(x[N*j+i]-0.5) + (y[N*j+i]-0.5)*(y[N*j+i]-0.5);
      f[N*j+i] = (r2-2*s2)/(r2*s2)*exp(-r2/(2*s2));
      u_a[N*j+i] = exp(-r2/(2*s2));
    }
  }

   k = new float[N];
  for (int i=0; i<=N/2; i++){
    k[i] = i * 2*M_PI;
  }
  for (int i=N/2+1; i<N; i++){
    k[i] = (i-N)* 2*M_PI;
  }

  cudaMalloc ((void**)&k_d, sizeof(float)*N);
  cudaMalloc ((void**)&f_d, sizeof(float)*N*N);
  cudaMalloc ((void**)&u_d, sizeof(float)*N*N);

  cudaEventRecord(start);
  cudaMemcpy(k_d, k, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(f_d, f, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  cudaMalloc ((void**)&ft_d, sizeof(cufftComplex)*N*N);
  cudaMalloc ((void**)&ft_d_k, sizeof(cufftComplex)*N*N);
  cudaMalloc ((void**)&f_dc, sizeof(cufftComplex)*N*N);
  cudaMalloc ((void**)&u_dc, sizeof(cufftComplex)*N*N);

  dim3 dimGrid2 (int(  (N-0.5)/BSZ )+1, int(  (N-0.5)/BSZ )+1,1  );
  dim3 dimBlock2 (BSZ, BSZ,1);
  real2complex<<<dimGrid2, dimBlock2>>>(f_d, f_dc, N);

  cufftPlan2d(&plan, N, N, CUFFT_C2C);
  cufftExecC2C(plan, f_dc, ft_d, CUFFT_FORWARD);
  complex2real<<<dimGrid, dimBlock>>>(u_dc, u_d, N);

  cudaMemcpy(u, u_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("tempo 256x256: %0.3fms\n",milliseconds);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);


  N = 512;
  xmax = 1.0f;
  xmin = 0.0f;
  ymin = 0.0f;
  h = (xmax-xmin)/((float)N);
  s = 0.1f;
  s2 = s*s;

  x = new float[N*N];
  y = new float[N*N];
  u = new float[N*N];
  f = new float[N*N];
  u_a = new float[N*N];
  err = new float[N*N];


  for(int j=0; j<N; j++){
    for(int i=0; i<N; i++){
      x[N*j+i] = xmin + i*h;
      y[N*j+i] = ymin + j*h;

      r2 = (x[N*j+i]-0.5)*(x[N*j+i]-0.5) + (y[N*j+i]-0.5)*(y[N*j+i]-0.5);
      f[N*j+i] = (r2-2*s2)/(r2*s2)*exp(-r2/(2*s2));
      u_a[N*j+i] = exp(-r2/(2*s2));
    }
  }

  k = new float[N];
  for (int i=0; i<=N/2; i++){
    k[i] = i * 2*M_PI;
  }
  for (int i=N/2+1; i<N; i++){
    k[i] = (i-N)* 2*M_PI;
  }

  cudaMalloc ((void**)&k_d, sizeof(float)*N);
  cudaMalloc ((void**)&f_d, sizeof(float)*N*N);
  cudaMalloc ((void**)&u_d, sizeof(float)*N*N);

  cudaEventRecord(start);
  cudaMemcpy(k_d, k, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(f_d, f, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  cudaMalloc ((void**)&ft_d, sizeof(cufftComplex)*N*N);
  cudaMalloc ((void**)&ft_d_k, sizeof(cufftComplex)*N*N);
  cudaMalloc ((void**)&f_dc, sizeof(cufftComplex)*N*N);
  cudaMalloc ((void**)&u_dc, sizeof(cufftComplex)*N*N);

  dim3 dimGrid3 (int(  (N-0.5)/BSZ )+1, int(  (N-0.5)/BSZ )+1,1  );
  dim3 dimBlock3 (BSZ, BSZ,1);
  real2complex<<<dimGrid3, dimBlock3>>>(f_d, f_dc, N);

  cufftPlan2d(&plan, N, N, CUFFT_C2C);
  cufftExecC2C(plan, f_dc, ft_d, CUFFT_FORWARD);
  complex2real<<<dimGrid, dimBlock>>>(u_dc, u_d, N);

  cudaMemcpy(u, u_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("tempo 512x512: %0.3fms\n",milliseconds);

  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  N = 1024;
  xmax = 1.0f;
  xmin = 0.0f;
  ymin = 0.0f;
  h = (xmax-xmin)/((float)N);
  s = 0.1f;
  s2 = s*s;

  x = new float[N*N];
  y = new float[N*N];
  u = new float[N*N];
  f = new float[N*N];
  u_a = new float[N*N];
  err = new float[N*N];


  for(int j=0; j<N; j++){
    for(int i=0; i<N; i++){
      x[N*j+i] = xmin + i*h;
      y[N*j+i] = ymin + j*h;

      r2 = (x[N*j+i]-0.5)*(x[N*j+i]-0.5) + (y[N*j+i]-0.5)*(y[N*j+i]-0.5);
      f[N*j+i] = (r2-2*s2)/(r2*s2)*exp(-r2/(2*s2));
      u_a[N*j+i] = exp(-r2/(2*s2));
    }
  }

  k = new float[N];
  for (int i=0; i<=N/2; i++){
    k[i] = i * 2*M_PI;
  }
  for (int i=N/2+1; i<N; i++){
    k[i] = (i-N)* 2*M_PI;
  }

  cudaMalloc ((void**)&k_d, sizeof(float)*N);
  cudaMalloc ((void**)&f_d, sizeof(float)*N*N);
  cudaMalloc ((void**)&u_d, sizeof(float)*N*N);

  cudaEventRecord(start);
  cudaMemcpy(k_d, k, sizeof(float)*N, cudaMemcpyHostToDevice);
  cudaMemcpy(f_d, f, sizeof(float)*N*N, cudaMemcpyHostToDevice);

  cudaMalloc ((void**)&ft_d, sizeof(cufftComplex)*N*N);
  cudaMalloc ((void**)&ft_d_k, sizeof(cufftComplex)*N*N);
  cudaMalloc ((void**)&f_dc, sizeof(cufftComplex)*N*N);
  cudaMalloc ((void**)&u_dc, sizeof(cufftComplex)*N*N);

  dim3 dimGrid4 (int(  (N-0.5)/BSZ )+1, int(  (N-0.5)/BSZ )+1,1  );
  dim3 dimBlock4 (BSZ, BSZ,1);
  real2complex<<<dimGrid4, dimBlock4>>>(f_d, f_dc, N);

  cufftPlan2d(&plan, N, N, CUFFT_C2C);
  cufftExecC2C(plan, f_dc, ft_d, CUFFT_FORWARD);
  complex2real<<<dimGrid, dimBlock>>>(u_dc, u_d, N);

  cudaMemcpy(u, u_d, sizeof(float)*N*N, cudaMemcpyDeviceToHost);

  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("tempo 1024x1024: %0.3fms\n",milliseconds);


  return 0;
}
