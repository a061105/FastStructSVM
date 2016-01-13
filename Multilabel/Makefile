CXX ?= g++
CC ?= gcc
CFLAGS = -w -fopenmp -O3
LIBS = blas/blas.a

all: clean train

train: CG.o blas/blas.a
	$(CXX) $(CFLAGS) -o train train.cpp CG.o $(LIBS)

CG.o: $(LIBS)
	        $(CXX) $(CFLAGS) -c -o CG.o CG.cpp

blas/blas.a: blas/*.c blas/*.h
	        make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';
clean:
		rm -f *~ *.o train
