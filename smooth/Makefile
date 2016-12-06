
CC = /usr/local/cuda/bin/nvcc
CFLAGS = 
OBJS =
BINSERIAL = mainSerial
BINPARALELO = mainParalelo

serial: serial.cu
	$(CC) -o $(BINSERIAL) $(CFLAGS)  serial.cu -lm

paralelo: smooth.cu
	$(CC) -o $(BINPARALELO) $(CFLAGS) smooth.cu -lm

clean:
	rm -f *.o *.out main
