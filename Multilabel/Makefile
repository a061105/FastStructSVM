CXX ?= g++
CC ?= gcc
CFLAGS = -w -fopenmp -O3 -std=c++11
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

scale=train
data_dir=/scratch/cluster/ianyen/data/
s=1

bibtex:
	./train -s $(s) -m 10000 $(data_dir)/multilabel/bibtex.$(scale)

yeast:
	./train -s $(s) -m 10000 $(data_dir)/multilabel/yeast_$(scale).svm
