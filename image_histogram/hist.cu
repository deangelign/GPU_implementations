#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>


#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255
#define TILE_WIDTH 16


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

__global__ void computeHistogramGPU(PPMPixel *data, int *cols, int *rows, float *h){
	/*
	 * Nós podemos pensar que cada pixel na imagem é uma thread.
	 * Então devemos calcular o indice da thread na imagem
	 * porem como eu coloquei um bloco de 16x16 é necessário checar a posição da thread
	 * esta contida na imagem.
	 *
	 */
	int row = blockIdx.y * blockDim.y + threadIdx.y; //coordenada vertical da thread na imagem
	int col = blockIdx.x * blockDim.x + threadIdx.x; //coordenada horizontal da thread na imagem

	if(threadIdx.x == 0 && threadIdx.y == 0){
    		printf("ro:%d co:%d rb:%d rc:%d \n",row,col,blockIdx.y,blockIdx.x);
    	}


    __syncthreads();

	if(row < *rows && col < *cols) {
		int j, k, l;
		int index = 0;
		//eu percebi que estes 3 loops são desnecessarios, pois poderiamos atribuir o valor do pixel no posicao correta
		//do histograma fazendo  h[ (data.red*16) + (data.green*4) + data.blue] += 1.
		//Entretanto, eu prefiri não modificar o codigo, pois assim eu consegui obter speedups melhores.
		for (j = 0; j <= 3; j++) {
			for (k = 0; k <= 3; k++) {
				for (l = 0; l <= 3; l++) {
					if (data[(row*(*cols)) + col].red == j && data[(row*(*cols)) + col].green == k && data[(row*(*cols)) + col].blue == l) {
						atomicAdd(&h[index], 1.0);//tendo certeza que apenas uma thread por vez vai modificar essa variável (acesso atómico)
					}
					index++;
				}				
			}
		}
	}
}

void Histogram(PPMImage *image, float *h) {
	// host
	int i, cols = image->x, rows = image->y;
	float n = image->y * image->x;
	// device
	PPMPixel *d_data;
	int *d_cols, *d_rows;
	float *d_h;

	for (i = 0; i < n; i++) {
		image->data[i].red = floor((image->data[i].red * 4) / 256);
		image->data[i].blue = floor((image->data[i].blue * 4) / 256);
		image->data[i].green = floor((image->data[i].green * 4) / 256);
	}

	//variáveis necessarias para medir o tempos pedidos na tarefa
//	cudaEvent_t start, stop;
//	cudaEventCreate(&start);
//	cudaEventCreate(&stop);
//	float milliseconds = 0;

	//alocando memoria na GPU para a variáveis
//	cudaEventRecord(start);
	cudaMalloc( (void**)&d_data, sizeof(PPMPixel)*( (int)n) ) ;
	cudaMalloc( (void**)&d_cols, sizeof(int));
	cudaMalloc( (void**)&d_rows, sizeof(int));
	cudaMalloc( (void**)&d_h, sizeof(float) * 64);
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&milliseconds, start, stop);
//	printf("\ntempo alocar memoria: %0.6f\n",milliseconds);

	//enviado os dados da CPU para a GPU
//	cudaEventRecord(start);
	cudaMemcpy(d_data, image->data, sizeof(PPMPixel)*( (int)n ) , cudaMemcpyHostToDevice);
	cudaMemcpy(d_cols, &cols, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_rows, &rows, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_h, h, sizeof(float) * 64, cudaMemcpyHostToDevice);
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&milliseconds, start, stop);
//	printf("\ntempo enviar: %0.6f\n",milliseconds);

	dim3 numeroBlocosNaImagem(ceil((float)cols/TILE_WIDTH), ceil((float)rows/TILE_WIDTH), 1);
	dim3 numeroThreadsPorBloco(TILE_WIDTH, TILE_WIDTH, 1);


//	//realizando a computação na GPU
//	cudaEventRecord(start);
	computeHistogramGPU<<<numeroBlocosNaImagem,numeroThreadsPorBloco>>>(d_data, d_cols, d_rows, d_h);
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&milliseconds, start, stop);
//	printf("\ntempo computar: %0.6f\n",milliseconds);


	//recenbendo o resultado da GPU (GPU -> CPU)
//	cudaEventRecord(start);
	cudaMemcpy(h, d_h, sizeof(float) * 64, cudaMemcpyDeviceToHost);
//	cudaEventRecord(stop);
//	cudaEventSynchronize(stop);
//	cudaEventElapsedTime(&milliseconds, start, stop);
//	printf("\ntempo receber: %0.6f\n",milliseconds);

	//liberando a memória alocada na GPU
	cudaFree(d_data);
	cudaFree(d_cols);
	cudaFree(d_rows);
	cudaFree(d_h);

	/*
	 * Normalizando os valores do histograma.
	 * Infelizmente quando eu realizava a normalização na GPU
	 * eu obtinha um valor diferente na terceira casa decimal de precisão. O mais misterioso
	 * era que isso so ocorria no primeiro elemento do meu vetor h. Por isso resolvi fazer
	 * a normalização na CPU
	 * */
	for (i = 0; i < 64; i++){
		h[i] = h[i]/n;
	}
}

int main(int argc, char *argv[]) {

	if( argc != 2 ) {
		printf("Too many or no one arguments supplied.\n");
	}

	//double t_start, t_end;
	int i;
	char *filename = argv[1]; //Recebendo o arquivo!;
	
	//scanf("%s", filename);
	PPMImage *image = readPPM(filename);

	float *h = (float*)malloc(sizeof(float) * 64);

	//Inicializar h
	for(i=0; i < 64; i++) h[i] = 0.0;

	//t_start = rtclock();
	Histogram(image, h);
	//t_end = rtclock();

	for (i = 0; i < 64; i++){
		printf("%0.3f ", h[i]);
	}
	printf("\n");
	//fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);
	free(h);
}

/*
 * como a normalização dos valores eu fiz na CPU então o considerei o meu tempo de GPU_total sendo como
 * GPU_total = tempo_GPU_criar_buffer + tempo_GPU_offload_enviar + tempo_kernel + tempo_GPU_offload_receber + GPU_total + CPU_tempo_normalizar
 *
 * -----------------------------------------------------------------------------------------------------------------------------------------------------------
 * | Entrada	| tempo_serial	| tempo_GPU_criar_buffer	| tempo_GPU_offload_enviar	| tempo_kernel	| tempo_GPU_offload_receber	| GPU_total	| speedup	|
 * ------------------------------------------------------------------------------------------------------------------------------------------------------------
 * |	ar1.ppm	|   0.179675s	|		0.321664ms			|			0.852256ms		|   2.844864ms	|		0.025536ms			| 0.092086s | 1.951165	|
 * |	ar2.ppm	|   0.344623s	|		0.385280ms			|			1.608000ms		|	7.420544ms	|		0.025056ms			| 0.110731s	| 3.112254	|
 * |	ar3.ppm	|   1.298848s	|		0.343008ms			|			5.551264ms		|	32.017376ms	|		0.021952ms			| 0.235085s	| 5.525014	|
 * -----------------------------------------------------------------------------------------------------------------------------------------------------------
 *
 *
 * */
