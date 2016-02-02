DirsMultilabel=$(wildcard Multilabel/*) #rcv1_regions bibtex Eur-Lex-Large
DirsChain=$(wildcard Chain/*) #POS speech ChineseOCR
Solvers=SSG BCFW Soft-BCFW Soft-BCFW-accurate GDMM GDMM-subFMO


all:
	for i in $(DirsChain); do \
		for j in $(Solvers); do \
			echo $$i/$$j.log ;  \
		done \
	done
	for i in $(DirsMultilabel); do \
		for j in $(Solvers); do \
			echo $$i/$$j.log ;  \
		done \
	done

.PHONY: $(wildcard Chain/*) $(wildcard Multilabel/*)

Chain/POS:
	./plotFile.sh POS $@/time_obj/*
