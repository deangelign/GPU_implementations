#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>

#define MASK_WIDTH 5
#define TILE_WIDTH 16

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255


typedef struct {
    unsigned char red, green, blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


static PPMImage *readPPM(const char *filename) {
    char buff[16];
    PPMImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }

    img = (PPMImage *) malloc(sizeof(PPMImage));
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n')
            ;
        c = getc(fp);
    }

    ungetc(c, fp);
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        exit(1);
    }

    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
                filename);
        exit(1);
    }

    if (rgb_comp_color != RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        exit(1);
    }

    while (fgetc(fp) != '\n')
        ;
    img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(PPMImage *img) {
    fprintf(stdout, "P6\n");
    fprintf(stdout, "# %s\n", COMMENT);
    fprintf(stdout, "%d %d\n", img->x, img->y);
    fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);
    fwrite(img->data, 3 * img->x, img->y, stdout);
    fclose(stdout);
}

//versao sem shared memory
__global__ void smooth_GPU_NoSM(PPMPixel *dataIn, PPMPixel *dataOut,  int *width, int *height) {


    //numeros de pixels que o bloco precisa expandir em cada direção (esquerda, direita, baixo, cima);
    int sizeExtraRegion = (int)((MASK_WIDTH - 1) / 2);

    //variaveis necessarias para criar uma regiao de dimensao GridSize x GridSize (shared memory)
    int i, j, red, green, blue;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;



    if(row < *height && col < *width){
        red = 0;
        green = 0;
        blue = 0;

        int rowInNeighbourhood;
        int colInNeighbourhood;
        int indexNeighbourhood;
        int numberPixelsInNeighbourhood = MASK_WIDTH * MASK_WIDTH;

        //pega a vizinhanca do pixel
        for(i = -sizeExtraRegion; i <= sizeExtraRegion; i++){
            for(j = -sizeExtraRegion; j <= sizeExtraRegion; j++){
                rowInNeighbourhood = row - i;
                colInNeighbourhood = col - j;
                indexNeighbourhood = (rowInNeighbourhood * *width) + colInNeighbourhood;

                if(rowInNeighbourhood >= 0 && rowInNeighbourhood < *height){
                    if(colInNeighbourhood >= 0 && colInNeighbourhood < *width){
                        red += dataIn[indexNeighbourhood].red;
                        green += dataIn[indexNeighbourhood].green;
                        blue += dataIn[indexNeighbourhood].blue;
                    }
                }
            }
        }


        dataOut[(row * *width) + col].red = red / (numberPixelsInNeighbourhood);
        dataOut[(row * *width) + col].green = green / (numberPixelsInNeighbourhood);
        dataOut[(row * *width) + col].blue = blue / (numberPixelsInNeighbourhood);
    }

}


__global__ void smooth_GPU(PPMPixel *dataIn, PPMPixel *dataOut,  int *width, int *height) {


    //numeros de pixels que o bloco precisa expandir em cada direção (esquerda, direita, baixo, cima);
    int sizeExtraRegion = (int)((MASK_WIDTH - 1) / 2);

    //variaveis necessarias para criar uma regiao de dimensao GridSize x GridSize (shared memory)
    int GridSize = TILE_WIDTH + (MASK_WIDTH - 1);
    int gridratio = ceil((float)(GridSize * GridSize) / (blockDim.x * blockDim.y));
    int startBlockRow = blockIdx.y * blockDim.y;
    int startBlockCol = blockIdx.x * blockDim.x;
    int it, index, x, y;
    int sharedRow, sharedCol;
    int i, j, red, green, blue;
    __shared__ PPMPixel shared[TILE_WIDTH + (MASK_WIDTH - 1)][TILE_WIDTH + (MASK_WIDTH - 1)];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    //transferindo os dados do bloco para a shared memory
    for(it = 0; it < gridratio; it++){
        index = (threadIdx.y * blockDim.x) + threadIdx.x + (it * blockDim.x * blockDim.y);
        x = (index / GridSize );
        y = index % GridSize ;

        if(x < GridSize  && y < GridSize ){
            sharedRow = startBlockRow + x - sizeExtraRegion ;
            sharedCol = startBlockCol + y - sizeExtraRegion ;
            if(sharedRow >= 0 && sharedCol >= 0 && sharedCol < *width && sharedRow < *height){
                shared[x][y] = dataIn[sharedRow * *width + sharedCol];
            }else{
                shared[x][y].red = shared[x][y].green = shared[x][y].blue = 0;
            }
        }
    }

    //sincronização necessaria para garantir que todos os dados estejam na shared memory antes de computar o smooth
    __syncthreads();


    //se é um pixel valido, computa o smooth
    if((row * col) < (*width * *height)){
        red = 0;
        green = 0;
        blue = 0;
        int numberPixelsInNeighbourhood = MASK_WIDTH * MASK_WIDTH;
        //pega a vizinhanca do pixel
        for(i = threadIdx.y; i < (threadIdx.y + MASK_WIDTH); i++){
            for(j = threadIdx.x; j < (threadIdx.x + MASK_WIDTH); j++){
                red += shared[i][j].red;
                green += shared[i][j].green;
                blue += shared[i][j].blue;
            }
        }
        dataOut[(row * *width) + col].red = red / (numberPixelsInNeighbourhood);
        dataOut[(row * *width) + col].green = green / (numberPixelsInNeighbourhood);
        dataOut[(row * *width) + col].blue = blue / (numberPixelsInNeighbourhood);
    }
}

int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }

    //double t_start, t_end;
    char *filename = argv[1];
    float milliseconds;

    // CUDA EVENTS
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    PPMImage *image = readPPM(filename);
    PPMImage *image_output = (PPMImage *) malloc(sizeof(PPMImage));
    image_output->x = image->x;
    image_output->y = image->y;
    image_output->data = (PPMPixel*) malloc(image_output->x * image_output->y * sizeof(PPMPixel));

    int w = image->x;
    int h = image->y;
    int n = image->y * image->x;

    PPMPixel *D_dataIn, *D_dataOut;
    int *D_width, *D_height;

    cudaEventRecord(start);
    cudaMalloc((void**)&D_dataIn, n * sizeof(PPMPixel));
    cudaMalloc((void**)&D_dataOut, n * sizeof(PPMPixel));
    cudaMalloc((void**)&D_width, sizeof(int));
    cudaMalloc((void**)&D_height, sizeof(int));

    cudaMemcpy(D_dataIn, image->data, n * sizeof(PPMPixel), cudaMemcpyHostToDevice);
    cudaMemcpy(D_width, &w, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(D_height, &h, sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPorBloco(TILE_WIDTH, TILE_WIDTH);
    dim3 blocos(ceil((float)w / TILE_WIDTH), ceil((float)h / TILE_WIDTH), 1);

    //cudaEventRecord(start);
    smooth_GPU<<<blocos, threadsPorBloco>>>(D_dataIn, D_dataOut, D_width, D_height);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(image_output->data, D_dataOut, n * sizeof(PPMPixel), cudaMemcpyDeviceToHost);

    cudaFree(D_dataIn);
    cudaFree(D_dataOut);
    cudaFree(D_width);
    cudaFree(D_height);

    // t_start = rtclock();
    // Smoothing_CPU_Serial(image_output, image);
    // t_end = rtclock();

    //writePPM(image_output);

    //fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);
    printf("Tempo: %0.3f\n",milliseconds);
    free(image->data);
    free(image);
    free(image_output->data);
    free(image_output);
    return 0;
}


/*
 *
 *
 * O cálculo usado para computar os valores da tabela foi
 *
 *
 * Bwr = (N_sm)/(N_gm)
 *
 * onde
 * Bwr: Bandwidth Reduction
 * N_sm: Número de acessos a shared memory
 * N_gm: Número de acessos a memoria global
 *
 * Na prática eu não estou usando uma matrix (ou vetor) para representar a máscara. Para realizar a computação
 * eu apenas preciso do valor do block extendido (shared memory). Exemplo, se o bloco na GPU tem tamaho 16x16 e a máscara é uma 5x5
 * então o tamanho da sharded memory é 20x20. A mémoria global é apenas acessada para transferir os dados para a shared
 * memory. Entao
 *
 * N_gm = 20*20 = 400
 *
 * Para computar a intensidade de um pixel no bloco, nós precisamos de 5x5 acessos a SM (shared memory).
 * como o bloco tem 16x16 pixels em um bloco, então temos
 *
 * N_sm = 16*16*5*5
 *
 * Bwr = (16*16*5*5)/(20*20)  = 16;
 *
 *
 * Abaixo está o Bandwidth Reduction para um único bloco.
 *-----------------------------------------------------------------------------------------------------------------------------
 *|                  | BLOCK_SIZE =  8x8  | BLOCK_SIZE = 14x14 | BLOCK_SIZE = 15x15 | BLOCK_SIZE = 16x16 | BLOCK_SIZE = 32x32 |
 *|------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
 *| MASK_WIDTH =   5 |              11.11 |              15.12 |              15.58 |              16.00 |              19.75 |
 *| MASK_WIDTH =   7 |              16.00 |              24.01 |              25.00 |              25.91 |              34.76 |
 *| MASK_WIDTH =   9 |              20.25 |              32.80 |              34.45 |              36.00 |              51.84 |
 *| MASK_WIDTH =  11 |              23.90 |              41.17 |              43.56 |              45.82 |              70.24 |
 *| MASK_WIDTH =  13 |              27.04 |              49.00 |              52.16 |              55.18 |              89.38 |
 * -----------------------------------------------------------------------------------------------------------------------------
 *
 *
 * Desta maneria, para saber o total de Bandwidth Reduction na imagem, precisamos saber o número de blocos gerados para
 * uma dada entrada. Por exempĺo, para a entrada arq3.ppm que tem dimensões 3840x2160 (2160 linhas e 3840 colunas),
 * utilizando um bloco de 16x16, temos ceil(3840/16) = 240 blocos para cada linha e ceil(2160/16) = 135 blocos em cada
 * coluna. Portanto o número total de blocos é de 240*135 = 32400. Multiplicando esse valor em cada célula da tabela acima
 * obtemos
 *
 *-----------------------------------------------------------------------------------------------------------------------------
 *|                  | BLOCK_SIZE =  8x8  | BLOCK_SIZE = 14x14 | BLOCK_SIZE = 15x15 | BLOCK_SIZE = 16x16 | BLOCK_SIZE = 32x32 |
 *|------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
 *| MASK_WIDTH =   5 |              359964|              489888|              504792|              518400|              639900|
 *| MASK_WIDTH =   7 |              518400|              777924|              810000|              839484|             1126224|
 *| MASK_WIDTH =   9 |              656100|             1062720|             1116180|             1166400|             1679616|
 *| MASK_WIDTH =  11 |              774360|             1333908|             1411344|             1484568|            2275776 |
 *| MASK_WIDTH =  13 |              876096|             1587600|             1689984|             1787832|            2895912 |
 * -----------------------------------------------------------------------------------------------------------------------------
 *
 *
 *
 * Conclusão: da maneira que eu implementei o meu smooth, o Bandwidth Reduction é proporcional ao tamanho
 * da máscara e ao tamanho do bloco. Abaixo esta os tempos e os speeds ups obtidos para cada uma das imagens considerando um bloco de
 * 16x16 e uma máscara de 5x5.
 *
 *
 *Abaxo segue as medicoes e speed ups obtidos
 * ---------------------------------------------------------------------------------------------------------
 * |Entrada  |CPU_Serial (ms)  |GPU_SharedMemory (ms)  |GPU_NoSharedMemory (ms)  |Speedup(CPU/GPUSM)       |
 * ---------------------------------------------------------------------------------------------------------
 * |arq1.ppm |160.054          | 0.416                 |1.096                    |384.74                   |
 * |arq2.ppm |361.446          | 0.878                 |2.421                    |411.67                   |
 * |arq3.ppm |1420.776         | 3.310                 |9.533                    |429.24                   |
 * ---------------------------------------------------------------------------------------------------------
 *
 * A tabela abaixo foi o tempo em ms (milisegundos) obtidos para a ar3.ppm (apenas), variando o tamanho do bloco
 * e o tamanho da máscara.
 *
 * ------------------------------------------------------------------------------------------------------
 * |             |BLOCK_SIZE=8x8   |BLOCK_SIZE=14x14|BLOCK_SIZE=15x15|BLOCK_SIZE=16x16|BLOCK_SIZE=32x32 |
 * ------------------------------------------------------------------------------------------------------
 * |MASK_WIDTH=5 |      5.734      |     3.954       |     3.816     |    3.310       |   3.517         |
 * |MASK_WIDTH=7 |      8.806      |     7.194       |     7.263     |    5.125       |   5.067         |
 * |MASK_WIDTH=9 |      11.150     |     11.454      |     11.151    |    7.371       |   7.014         |
 * |MASK_WIDTH=11|      18.258     |     16.342      |     15.701    |   10.347       |   9.673         |
 * |MASK_WIDTH=13|      25.179     |     21.975      |     21.018    |   18.684       |   12.710        |
 * ------------------------------------------------------------------------------------------------------
 *
 * Conclusao: O tamanho da mascara para computar o valor do pixel afeta consideravelmente o tempo de execução do
 * programa.
 *
 * */
