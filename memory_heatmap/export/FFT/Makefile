OBJ = main.o fftmisc.o  fourierf.o
FILE = main.c fftmisc.c  fourierf.c
CFLAGS = -static -O3 -g
CXX = gcc
ACC = arm-linux-gnueabi-gcc

fft: ${OBJ} Makefile
	${CXX}  ${CFLAGS} ${OBJ} -o fft -lm
fftmisc.o: fftmisc.c
	${CXX} ${CFLAGS} -c fftmisc.c
fourierf.o: fourierf.c
	${CXX} ${CFLAGS} -c fourierf.c
main.o: main.c
	${CXX} ${CFLAGS} -c main.c

clean:
	rm -rf *.o fft output*
