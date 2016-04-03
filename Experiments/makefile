DirsMultilabel=$(wildcard Multilabel/*) #rcv1_regions bibtex Eur-Lex-Large
DirsChain=$(wildcard Chain/*) #POS speech ChineseOCR
Solvers=SSG BCFW Soft-BCFW Soft-BCFW-accurate GDMM GDMM-subFMO


all:
	for i in $(DirsMultilabel); do \
		make $$i; \
	done
	for i in $(DirsChain); do \
		make $$i; \
	done

.PHONY: $(DirsChain) $(DirsMultilabel)

Chain/POS:
	rm -f $@/time_obj/*
	rm -f $@/time_acc/*
	$(MAKE) -C $@ time_obj
	$(MAKE) -C $@ time_acc
	./plotAcc.sh $(@F)-Acc plot$(@F)Acc $@/time_acc/*
	#./plotObj.sh $(@F)-Obj plot$(@F)Obj $@/time_obj/*
	#./plotAcc.sh $(@F)-Acc plot$(@F)Acc $@/iter_acc/*
	#./plotObj.sh $(@F)-Obj plot$(@F)Obj $@/iter_obj/*


Chain/speech:
	#rm -f $@/time_obj/*
	#rm -f $@/time_acc/*
	#$(MAKE) -C $@ time_obj
	#$(MAKE) -C $@ time_acc
	#./plotAcc.sh $(@F)-Acc plot$(@F)Acc $@/time_acc/*
	#./plotObj.sh $(@F)-Obj plot$(@F)Obj $@/time_obj/*

Chain/ChineseOCR:
	rm -f $@/time_obj/*
	rm -f $@/time_acc/*
	$(MAKE) -C $@ time_obj
	$(MAKE) -C $@ time_acc
	./plotAcc.sh $(@F)-Acc plot$(@F)Acc $@/time_acc/*
	./plotObj.sh $(@F)-Obj plot$(@F)Obj $@/time_obj/*

Multilabel/bibtex:
	#rm -f $@/time_obj/*
	#rm -f $@/time_acc/*
	#$(MAKE) -C $@ time_obj
	#$(MAKE) -C $@ time_acc
	#./plotAcc.sh $(@F)-Acc plot$(@F)Acc $@/time_acc/*
	#./plotObj.sh $(@F)-Obj plot$(@F)Obj $@/time_obj/*

Multilabel/rcv1:
	rm -f $@/time_obj/*
	rm -f $@/time_acc/*
	$(MAKE) -C $@ time_obj
	$(MAKE) -C $@ time_acc
	./plotAcc.sh $(@F)-Acc plot$(@F)Acc $@/time_acc/*
	./plotObj.sh $(@F)-Obj plot$(@F)Obj $@/time_obj/*

Multilabel/Eur-Lex:
	rm -f $@/time_obj/*
	rm -f $@/time_acc/*
	$(MAKE) -C $@ time_obj
	$(MAKE) -C $@ time_acc
	./plotAcc.sh $(@F)-Acc plotEurLexAcc $@/time_acc/*
	#./plotObj.sh $(@F)-Obj plotEurLexObj $@/time_obj/*
