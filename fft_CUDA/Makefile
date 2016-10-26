CC = nvcc
CFLAGS =
OBJS = UTIL/util.o SERIAL/fft.o GPU/fft.o
BIN = main

file: $(OBJS) main.o
	$(CC) -o $(BIN) $(CFLAGS) main.o $(OBJS) -lm
UTIL/util.o:
	$(CC) -c -o UTIL/util.o UTIL/util.cu
SERIAL/fft.o:
	$(CC) -c -o SERIAL/fft.o SERIAL/fft.cu
GPU/fft.o:
	$(CC) -c -o GPU/fft.o GPU/fft.cu
main.o:
	$(CC) -c -o main.o main.cu
run:
	./main > output.dat
clean:
	rm -rf  *.o **/*.o main output.dat OUTPUT/*.ppm
