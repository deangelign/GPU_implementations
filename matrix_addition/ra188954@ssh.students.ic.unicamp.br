#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_WIDTH 16

___global__ void addMatrix(int *A, int *B,int *C, int *rows, int *cols){

    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;

    if(col < (*cols) && row < (*rows)){
        C[ (row *  (*cols)) + col] = A[ (row *  (*cols)) + col] + B[ (row *  (*cols)) + col]  ;
    }

}

int main()
{
    int *A, *B, *C;
    int *D_A, *D_B, *D_C;
    int D_ROWS, D_COLS;
    int i, j;

    //Input
    int linhas, colunas;

    scanf("%d", &linhas);
    scanf("%d", &colunas);



    int size = sizeof(int) * linhas * colunas;


    //Alocando memória na CPU
    A = (int *)malloc(size);
    B = (int *)malloc(size);
    C = (int *)malloc(size);

    cudaMalloc( (void **)&D_A, size);
    cudaMalloc( (void **)&D_B, size);
    cudaMalloc( (void **)&D_C, size);

    cudaMalloc( (void **)&D_ROWS, sizeof(int));
    cudaMalloc( (void **)&D_COLS, sizeof(int));


    //Inicializar
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            A[i*colunas+j] =  B[i*colunas+j] = i+j;
        }
    }

    cudaMemcpy(D_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(D_B, B, size, cudaMemcpyHostToDevice);

    cudaMemcpy(D_ROWS, &linhas, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(D_COLS, &colunas, sizeof(int), cudaMemcpyHostToDevice);

    //Computacao que deverá ser movida para a GPU (que no momento é executada na CPU)
    //Lembrar que é necessário usar mapeamento 2D (visto em aula)
    dim3 numeroDeBlocosNaMatriz( ceil((float)cols/TILE_WIDTH),ceil((float)rows/TILE_WIDTH), 1 );
    dim3 numeroDeThreadsPorBloco( TILE_WIDTH,TILE_WIDTH, 1 );

    addMatrix<<<numeroDeBlocosNaMatriz,numeroDeThreadsPorBloco>>>(D_A,D_B,D_C, D_ROWS, D_COLS);
    cudaMemcpy(C, D_C, size, cudaMemcpyDeviceToHost);


    long long int somador=0;
    //Manter esta computação na CPU
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            somador+=C[i*colunas+j];   
        }
    }
    
    printf("%lli\n", somador);

    free(A);
    free(B);
    free(C);
    cudaFree(D_A);
    cudaFree(D_B);
    cudaFree(D_C);
}

