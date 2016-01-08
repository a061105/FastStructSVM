FLAG= -fopenmp -std=c++11 -O3 -w

all:
	g++ $(FLAG) -o train train.cpp

s=1
FLAG=
pos:
	$(eval train_file := /scratch/cluster/ianyen/data/POS/wsj.pos.crf.subtrain)
	$(eval model := model)
	./train -s $(s) $(FLAG) $(train_file) $(model) 

speech:
	$(eval train_file := /scratch/cluster/ianyen/data/speech/speech.subtrain)
	$(eval model := model)
	./train -s $(s) $(FLAG) $(train_file) $(model)
