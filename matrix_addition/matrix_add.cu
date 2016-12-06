#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//resolvi usar o mesmo tamanho de "bloco" que foi usado no exemplo do slide (slide 68, "CudaBasics")
#define TILE_WIDTH 16


__global__ void addMatrix(int *A, int *B,int *C, int *rows, int *cols){

    //variaveis necessarias para computar o indice corretamente
    int col = (blockDim.x * blockIdx.x) + threadIdx.x;
    int row = (blockDim.y * blockIdx.y) + threadIdx.y;

    /*
     * como alguma das dimensoes da matriz pode nao ser divisivel pelo o tamanho do bloco (TILE_WIDTH =16)
     * entao e existem indices em alguns blocos que estao mapeados em "nada" (não mapeam algum elemento na matriz).
     * Portanto é necessario esse condicional  para evitar acessos a posições de memorias indesejaveis.
     * */
    if(col < (*cols) && row < (*rows)){
        /*
         * A expressão
         *
         * (row *  (*cols)) + col
         *
         * computa o indice da corretamente
         * */
        C[ (row *  (*cols)) + col] = A[ (row *  (*cols)) + col] + B[ (row *  (*cols)) + col]  ;
    }

}

int main()
{
    //copias das variaveis da CPU
    int *A, *B, *C;

    //copias das variaveis da GPU
    int *D_A, *D_B, *D_C;
    int *D_ROWS, *D_COLS;


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

    //Alocando memória na GPU
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

    //enviando os dados para a GPU
    cudaMemcpy(D_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(D_B, B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(D_ROWS, &linhas, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(D_COLS, &colunas, sizeof(int), cudaMemcpyHostToDevice);

    //crieando as variaveis necessarias para fazer o mapeamento 2D
    dim3 numeroDeBlocosNaMatriz( ceil((float)colunas/TILE_WIDTH),ceil((float)linhas/TILE_WIDTH), 1 );
    dim3 numeroDeThreadsPorBloco( TILE_WIDTH,TILE_WIDTH, 1 );

    //executanto a soma de matriz na GPU
    addMatrix<<<numeroDeBlocosNaMatriz,numeroDeThreadsPorBloco>>>(D_A,D_B,D_C, D_ROWS, D_COLS);

    //transferindo o resultado da soma, que esta na GPU, para a CPU.
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
    //desalocando a memoria na GPU
    cudaFree(D_A);
    cudaFree(D_B);
    cudaFree(D_C);
    cudaFree(D_ROWS);
    cudaFree(D_COLS);

    return 0;
}

