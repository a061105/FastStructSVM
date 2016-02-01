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
data_dir=/scratch/cluster/ianyen/data/multilabel/
s=1
a=1
m=10000
e=10
t=100
o=10
q=1
opt=
r=-1
c=1.0

bibtex:
	./train -s $(s) -m 10000 $(data_dir)/multilabel/bibtex.$(scale)

yeast:
	$(eval train_file := $(data_dir)/yeast_$(scale).tmp)
	$(eval test_file := $(data_dir)/yeast_subtrain.tmp)
	./train -c $(c) -r $(r) -q $(q) $(opt) -o $(o) -t $(t) -s $(s) -a $(a) -m $(m) -e $(e) -h $(test_file) $(train_file)

rcv1_regions:
	$(eval train_file := $(data_dir)/rcv1_regions_$(scale).svm)
	$(eval test_file := $(data_dir)/rcv1_regions_test.03k)
	./train -o $(o) -t $(t) -s $(s) -a $(a) -m $(m) -e $(e) -h $(test_file) $(train_file)

Eur-Lex:
	$(eval train_file := $(data_dir)/Eur-Lex.$(scale))
	$(eval heldout_file := $(data_dir)/Eur-Lex.heldout)
	$(eval test_file := $(data_dir)/Eur-Lex.test)
	./train -s $(s) -a $(a) -m 10000 -h $(test_file) $(train_file)
	
