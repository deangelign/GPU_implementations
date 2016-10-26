#include <stdio.h>
#include <stdlib.h>

#include "util.h"

t_image* read_ppm_image(const char *filename){
  char buff[16];
  t_image *image;
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

  image = (t_image *) malloc(sizeof(t_image));
  if (!image) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  c = getc(fp);
  while (c == '#') {
    while (getc(fp) != '\n');
    c = getc(fp);
  }

  ungetc(c, fp);
  if (fscanf(fp, "%d %d", &image->y, &image->x) != 2) {
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

  while (fgetc(fp) != '\n');
  image->data = (t_pixel*) malloc(image->x * image->y * sizeof(t_pixel));

  if (!image) {
    fprintf(stderr, "Unable to allocate memory\n");
    exit(1);
  }

  if (fread(image->data, 3 * image->x, image->y, fp) != image->y) {
    fprintf(stderr, "Error loading image '%s'\n", filename);
    exit(1);
  }

  fclose(fp);
  return image;
}

void write_ppm_image(const char *filename, t_image *image){
  int i, j, index;
  FILE *fp = fopen(filename, "wb");
  (void) fprintf(fp, "P6\n%d %d\n%d\n", image->y, image->x, (int) RGB_COMPONENT_COLOR);
  for (i = 0; i < image->x; i++){
    for (j = 0; j < image->y; j++){
      index = (i * image->y) + j;
      static unsigned char color[3];
      color[0] = image->data[index].red;
      color[1] = image->data[index].green;
      color[2] = image->data[index].blue;
      (void) fwrite(color, 1, 3, fp);
    }
  }
  (void) fclose(fp);
}

void write_ppm_gray_image(const char *filename, t_gray_image *gray_image){
  int i, j;
  FILE *fp = fopen(filename, "wb");
  (void) fprintf(fp, "P6\n%d %d\n%d\n", gray_image->y, gray_image->x, (int) RGB_COMPONENT_COLOR);
  for (i = 0; i < gray_image->x; i++){
    for (j = 0; j < gray_image->y; j++){
      static unsigned char color[3];
      color[0] = color[1] = color[2] = gray_image->data[(i * gray_image->y) + j];
      (void) fwrite(color, 1, 3, fp);
    }
  }
  (void) fclose(fp);
}

void write_complex_image(const char *filename, t_complex_image *complex_image){
  int i, j, index;
  FILE *fp = fopen(filename, "wb");
  (void) fprintf(fp, "%d %d\n", complex_image->y, complex_image->x);
  for (i = 0; i < complex_image->x; i++){
    for (j = 0; j < complex_image->y; j++){
      index = (i * complex_image->y) + j;
      fprintf(fp, "(%.4f)(%.4f) /", complex_image->real[index], complex_image->imag[index]);
    }
    fprintf(fp, "\n");
  }
  (void) fclose(fp);
}

t_image* alloc_image(int rows, int columns){
  t_image *image = (t_image*)malloc(sizeof(t_image));
  image->x = rows;
  image->y = columns;
  image->data = (t_pixel *) malloc((rows * columns) * sizeof(t_pixel));
  int i, j, index;
  for (i = 0; i < rows; i++){
    for (j = 0; j < columns; j++){
      index = (i * columns) + j;
      image->data[index].red = image->data[index].green = image->data[index].blue = 0;
    }
  }
  return image;
}

t_gray_image* alloc_gray_image(int rows, int columns){
  t_gray_image *gray_image = (t_gray_image*)malloc(sizeof(t_gray_image));
  gray_image->x = rows;
  gray_image->y = columns;
  gray_image->data = (unsigned char *) malloc((rows * columns) * sizeof(unsigned char));
  int i, j;
  for (i = 0; i < rows; i++){
    for (j = 0; j < columns; j++){
      gray_image->data[(i * columns) + j] = 0;
    }
  }
  return gray_image;
}

t_complex_image* alloc_complex_image(int rows, int columns){
  t_complex_image *complex_image = (t_complex_image*)malloc(sizeof(t_complex_image));
  complex_image->x = rows;
  complex_image->y = columns;
  complex_image->real = (float *) malloc((complex_image->x * complex_image->y) * sizeof(float));
  complex_image->imag = (float *) malloc((complex_image->x * complex_image->y) * sizeof(float));
  int i, j;
  for (i = 0; i < rows; i++){
    for (j = 0; j < columns; j++){
      complex_image->real[(i * columns) + j] = complex_image->imag[(i * columns) + j] = 0.0;
    }
  }
  return complex_image;
}

void free_image(t_image *image){
  free(image->data);
  free(image);
}

void free_gray_image(t_gray_image *gray_image){
  free(gray_image->data);
  free(gray_image);
}

void free_complex_image(t_complex_image *complex_image){
  free(complex_image->real);
  free(complex_image->imag);
  free(complex_image);
}

t_gray_image* image_2_gray_image(t_image *image){
  t_gray_image *gray_image = alloc_gray_image(image->x, image->y);
  int i, j, index;
  float gray;
  float factor[] = { 0.2126, 0.7152, 0.0722 };
  for (i = 0; i < image->x; i++){
    for (j = 0; j < image->y; j++){ 
      index = (i * image->y) + j;
      gray = image->data[index].red * factor[0];
      gray += image->data[index].green * factor[1];
      gray += image->data[index].blue * factor[2];
      gray_image->data[index] = (unsigned char) gray;
    }
  }
  return gray_image;
}

t_complex_image* gray_image_2_complex_image(t_gray_image *gray_image){
  t_complex_image *complex_image = alloc_complex_image(gray_image->x, gray_image->y);
  int i, j, index;
  for (i = 0; i < gray_image->x; i++){
    for (j = 0; j < gray_image->y; j++){
      index = (i * gray_image->y) + j;
      complex_image->real[index] = (float)gray_image->data[index];
      complex_image->imag[index] = 0.0;
    }
  }
  return complex_image;
}

t_image* gray_image_2_image(t_gray_image *gray_image){
  t_image *image = alloc_image(gray_image->x, gray_image->y);
  int i, j, index;
  for (i = 0; i < image->x; i++){
    for (j = 0; j < image->y; j++){
      index = (i * image->y) + j;
      image->data[index].red = image->data[index].green = image->data[index].blue = gray_image->data[index];
    }
  }
  return image;
}

void shift_frequency_domain(t_complex_image *complex_image){
  int i, j, index;
  for (i = 0; i < complex_image->x; i++){
    for (j = 0; j < complex_image->y; j++){
      index = (i * complex_image->y) + j;
      complex_image->real[index] = ((i + j) % 2 == 0) ? complex_image->real[index] : -complex_image->real[index];
    }
  }
}

t_gray_image* image_spectre(t_complex_image *complex_image){
  t_gray_image *gray_image = alloc_gray_image(complex_image->x, complex_image->y);
  float *temp = (float*) malloc(sizeof(float) * complex_image->x * complex_image->y);
  int i, j, id;
  float max = -1;
  int count = 0;
  for (i = 0; i < complex_image->x; i++){
    for (j = 0; j < complex_image->y; j++){
      id = (i * complex_image->y) + j;
      temp[id] = log(sqrt((complex_image->real[id] * complex_image->real[id]) + (complex_image->imag[id] * complex_image->imag[id])) + 1);
      if(temp[id] > 0){count++;}
      max = (temp[id] > max) ? temp[id] : max;
    }
  }
  for (i = 0; i < complex_image->x; i++){
    for (j = 0; j < complex_image->y; j++){
      id = (i * complex_image->y) + j;
      gray_image->data[id] = (unsigned char)((temp[id] / max) * RGB_COMPONENT_COLOR);
    }
  }
  free(temp);
  return gray_image;
}