FLAG= -fopenmp -std=c++11 -O3 -w

all:
	g++ $(FLAG) -o train train.cpp

s=1
m=50
q=1
opt=
scale=train

pos:
	$(eval train_file := /scratch/cluster/ianyen/data/POS/wsj.pos.crf.$(scale))
	$(eval model := model)
	./train -s $(s) -q $(q) -m $(m) $(opt) $(train_file) $(model) 

speech:
	$(eval train_file := /scratch/cluster/ianyen/data/speech/speech.$(scale))
	$(eval model := model)
	./train -s $(s) -q $(q) -m $(m) $(opt) $(train_file) $(model)

OCR:
	$(eval train_file := /scratch/cluster/ianyen/data/ChineseOCR/data.fea.$(scale))
	$(eval model := model)
	./train -s $(s) -q $(q) -m $(m) $(opt) $(train_file) $(model)
