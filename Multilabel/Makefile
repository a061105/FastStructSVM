CXX=g++-4.8
CC=gcc-4.8
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
data_dir=.#/scratch/cluster/ianyen/data/multilabel/
s=1
a=1
m=500
e=3
t=0.1
o=500
q=3
opt=
r=-1
c=1.0

bibtex:
	./train -s $(s) -m 10000 $(data_dir)/multilabel/bibtex.$(scale)

yeast:
	$(eval train_file := $(data_dir)/yeast_$(scale).svm)
	$(eval test_file := $(data_dir)/yeast_test.svm)
	./train -c $(c) -r $(r) -q $(q) $(opt) -o $(o) -t $(t) -s $(s) -a $(a) -m $(m) -e $(e) -h $(test_file) $(train_file) "yeast.model"

rcv1_regions:
	$(eval train_file := $(data_dir)/rcv1_regions.train)
	$(eval test_file := $(data_dir)/rcv1_regions_test.03k)
	./train -c $(c) -r $(r) -q $(q) $(opt) -o $(o) -t $(t) -s $(s) -a $(a) -m $(m) -e $(e) -h $(test_file) $(train_file) "rcv1.model"

Eur-Lex:
	$(eval train_file := $(data_dir)/Eur-Lex.$(scale))
	$(eval heldout_file := $(data_dir)/Eur-Lex.heldout)
	$(eval test_file := $(data_dir)/Eur-Lex.test)
	./train -s $(s) -a $(a) -m 10000 -h $(test_file) $(train_file)
	
