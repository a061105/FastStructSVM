FLAG= -fopenmp -std=c++11 -O3 -w

all:
	g++ $(FLAG) -o train train.cpp

pos:
	$(eval train_file := /scratch/cluster/ianyen/data/POS/wsj.pos.crf.subtrain)
	$(eval model := model)
	./train $(train_file) $(model) 
