CC = nvcc
CFLAGS = 
OBJS = main.o
BIN = main

file: $(OBJS)
	$(CC) -o $(BIN) $(CFLAGS) $(OBJS) -lm -lcufft
main.o:
	$(CC) -c -o main.o main.cu
run:
	./main > output_cufft.dat
clean:
	rm -f *.o *.out main
