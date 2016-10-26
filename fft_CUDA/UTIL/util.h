#ifndef UTIL

#define UTIL
#define PI 3.14159265
#define RGB_COMPONENT_COLOR 255


typedef struct {
	float real, imag;
} t_complex;

typedef struct {
	int x, y;
	float *real, *imag;
} t_complex_image;

typedef struct {
	unsigned char red, green, blue;
} t_pixel;

typedef struct {
	int x, y;
	t_pixel *data;
} t_image;

typedef struct {
	int x, y;
	unsigned char *data;
} t_gray_image;

t_image* read_ppm_image(const char *filename);
void write_ppm_image(const char *filename, t_image *image);
void write_ppm_gray_image(const char *filename, t_gray_image *gray_image);
void write_complex_image(const char *filename, t_complex_image *complex_image);
t_image* alloc_image(int rows, int columns);
t_gray_image* alloc_gray_image(int rows, int columns);
t_complex_image* alloc_complex_image(int rows, int columns);
void free_image(t_image *image);
void free_gray_image(t_gray_image *gray_image);
void free_complex_image(t_complex_image *complex_image);
t_gray_image* image_2_gray_image(t_image *image);
t_complex_image* gray_image_2_complex_image(t_gray_image *gray_image);
t_image* gray_image_2_image(t_gray_image *gray_image);
void shift_frequency_domain(t_complex_image *complex_image);
t_gray_image* image_spectre(t_complex_image *complex_image);

#endif
