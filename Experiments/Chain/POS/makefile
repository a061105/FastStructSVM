dir_struct=../../../
data_dir=/scratch/cluster/ianyen/data
train_file=$(data_dir)/POS/wsj.pos.crf.subtrain2
test_file=$(data_dir)/POS/wsj.pos.crf.test

Solvers=SSG BCFW Soft-BCFW Soft-BCFW-accurate GDMM GDMM-subFMO GDMM-subFMO-acc

##all:
##	for i in $(Solvers); do \
##		make $$i.log ; \
##	done

.PRECIOUS: %.log

SSG.log: $(train_file) $(test_file)
	((./$(dir_struct)/ComparedSolvers/chain/structTrain $(train_file) $(test_file) 1.0 model.SSG 0 1000) 2>&1 ) >> SSG.log

BCFW.log: $(train_file) $(test_file)
	((./$(dir_struct)/ComparedSolvers/chain/structTrain $(train_file) $(test_file) 1.0 model.BCFW 1 1000) 2>&1 ) >> BCFW.log

Soft-BCFW.log: $(train_file) $(test_file)
	((./$(dir_struct)/ChainStruct/train -s 1 -b -u -q 1 -a 0 -t 1 -m 10000 -e 1000 -o 1 -h $(test_file) $(train_file) model.Soft-BCFW) 2>&1 ) >> Soft-BCFW.log

Soft-BCFW-accurate.log: $(train_file) $(test_file)
	((./$(dir_struct)/ChainStruct/train -s 1 -b -u -q 1 -a 0 -t 10 -m 10000 -e 1000 -o 1 -h $(test_file) $(train_file) model.Soft-BCFW-accurate) 2>&1 ) >> Soft-BCFW-accurate.log

GDMM.log: $(train_file) $(test_file)
	((./$(dir_struct)/ChainStruct/train -s 1 -b -q 1 -a 1 -t 0.1 -m 10000 -e 1000 -o 1 -h $(test_file) $(train_file) model.GDMM) 2>&1 ) >> GDMM.log

GDMM-subFMO.log: $(train_file) $(test_file) 
	((./$(dir_struct)/ChainStruct/train -s 1 -q 1 -a 1 -t 0.1 -m 10000 -e 1000 -o 1 -h $(test_file) $(train_file) model.GDMM-subFMO) 2>&1 ) >> GDMM-subFMO.log

GDMM-subFMO-acc.log: $(train_file) $(test_file) 
	((./$(dir_struct)/ChainStruct/train -s 1 -q 1 -a 1 -t 10 -m 10000 -e 1000 -o 1 -h $(test_file) $(train_file) model.GDMM-subFMO-acc) 2>&1 ) >> GDMM-subFMO-acc.log

.PHONY: time_obj time_acc iter_obj iter_acc

time_obj:
	for i in GDMM-subFMO SSG BCFW Soft-BCFW Soft-BCFW-accurate; do \
		make time_obj/$$i ; \
	done

time_obj/SSG:
	echo "time objective" >> time_obj/SSG
	cat SSG.log | grep 'time=' | sed 's/^.*time=//' | sed 's/,.*,/,/' | sed 's/,.*p-obj=/ /' >> time_obj/SSG

time_obj/BCFW: 
	echo "time objective" >> time_obj/BCFW
	cat BCFW.log | grep 'p-obj' | sed 's/^.*time=//' | sed 's/,.*p-obj=/ /' >> time_obj/BCFW 

time_obj/%:
	echo "time objective" >> time_obj/$*
	cat $*.log | grep 'p_obj' | sed 's/^.*p_obj=//' | sed 's/,.*overall time=/ /' | awk '{ print $$2 " " $$1; }' >> time_obj/$*

time_acc:
	for i in SSG BCFW Soft-BCFW Soft-BCFW-accurate GDMM-subFMO; do \
		make time_acc/$$i ; \
	done

time_acc/SSG:
	echo "time accuracy" >> time_acc/SSG
	cat SSG.log | grep 'test_acc=' | sed 's/^.*test_acc=//' | sed 's/,.*time=/ /' | sed "s/,.*$$//" | awk '{ print $$2 " " $$1; }' >> time_acc/SSG

time_acc/BCFW:	
	echo "time accuracy" >> time_acc/BCFW
	cat BCFW.log | grep 'test_acc=' | sed 's/^.*test_acc=//' | sed 's/,.*time=/ /' | sed 's/,.*$$//' | awk '{ print $$2 " " $$1; }' >> time_acc/BCFW

time_acc/%:
	echo "time accuracy" >> time_acc/$*
	cat $*.log | grep 'heldout Acc' | sed 's/^.*heldout Acc=//' | sed 's/,.*overall time=/ /' | sed 's/ (.*)//' | awk '{ print $$2 " " $$1; }' >> time_acc/$*

iter_obj:
	for i in Soft-BCFW Soft-BCFW-accurate GDMM; do \
                make iter_obj/$$i ; \
        done

iter_obj/GDMM:
	echo "iteration objective" >> $@
	cat $(@F).log | grep 'p_obj' | sed 's/^i=//' | sed 's/,.*p_obj=/ /' | sed 's/,.*$$//' >> iter_obj/$(@F)

iter_obj/GDMM-subFMO:
	echo "iteration objective" >> $@
	cat $(@F).log | grep 'p_obj' | sed 's/^i=//' | sed 's/,.*p_obj=/ /' | sed 's/,.*$$//' >> iter_obj/$(@F)
	
iter_obj/Soft-BCFW:
	echo "iteration objective" >> $@
	cat $(@F).log | grep 'p_obj' | sed 's/^i=//' | sed 's/,.*p_obj=/ /' | sed 's/,.*$$//' >> iter_obj/$(@F)

iter_obj/Soft-BCFW-accurate:
	echo "iteration objective" >> $@
	cat $(@F).log | grep 'p_obj' | sed 's/^i=//' | sed 's/,.*p_obj=/ /' | sed 's/,.*$$//' >> iter_obj/$(@F)	

iter_acc:
	for i in Soft-BCFW Soft-BCFW-accurate GDMM GDMM-subFMO; do \
                make iter_acc/$$i ; \
        done

iter_acc/GDMM:
	echo "iteration accuracy" >> $@
	cat $(@F).log | grep 'heldout Acc=' | sed 's/^i=//' | sed 's/,.*heldout Acc=/ /' | sed 's/ (.*)//' |  sed 's/,.*$$//' >> $@

iter_acc/GDMM-subFMO:
	echo "iteration accuracy" >> $@
	cat $(@F).log | grep 'heldout Acc=' | sed 's/^i=//' | sed 's/,.*heldout Acc=/ /' | sed 's/ (.*)//' |  sed 's/,.*$$//' >> $@
	
iter_acc/Soft-BCFW:
	echo "iteration accuracy" >> $@
	cat $(@F).log | grep 'heldout Acc=' | sed 's/^i=//' | sed 's/,.*heldout Acc=/ /' | sed 's/ (.*)//' |  sed 's/,.*$$//' >> $@

iter_acc/Soft-BCFW-accurate:
	echo "iteration accuracy" >> $@
	cat $(@F).log | grep 'heldout Acc=' | sed 's/^i=//' | sed 's/,.*heldout Acc=/ /' | sed 's/ (.*)//' |  sed 's/,.*$$//' >> $@
