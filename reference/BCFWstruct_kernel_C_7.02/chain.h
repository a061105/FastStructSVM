#ifndef CHAIN_H
#define CHAIN_H
#include "struct.h"
#include "kernel.h"
#include <iostream>

double chain_loss(Param* param, Label* ytrue, Label* ypred){ //assume of the same length
	
	int loss = 0;
	for(int i=0;i<ytrue->size();i++){
		if( ytrue->at(i) != ypred->at(i) )
			loss++;
	}
	return ((double)loss)/ytrue->size();
}

void chain_featuremap(Param* param, Instance* ins, Label* y, SparseVect* phi){
	
	int numVars = ins->num_vars;
	int numStates = param->num_states;
	
	phi->clear();
	for(int i=0;i<numVars-1;i++){
		int ind = numStates * y->at(i) + y->at(i+1);
		phi->push_back(make_pair(ind,1.0));
	}
	sumReduce(phi);
}

void chain_logDecode(double* nodePot, double* edgePot, int num_vars, int num_states, Label* y_decode){
	
	double** max_marg = new double*[num_vars];
	int** argmax = new int*[num_vars];
	for(int i=0;i<num_vars;i++){
		max_marg[i] = new double[num_states];
		argmax[i] = new int[num_states];
		for(int j=0;j<num_states;j++){
			max_marg[i][j] = -1e300;
			argmax[i][j] = -1;
		}
	}
	
	//initialize
	for(int i=0;i<num_states;i++)
		max_marg[0][i] = nodePot[0*num_states+i];
	//forward pass
	for(int t=1;t<num_vars;t++){
		for(int i=0;i<num_states;i++){
			int ind = i*num_states;
			for(int j=0;j<num_states;j++){
				double new_val =  max_marg[t-1][i] + edgePot[ind+j];
				if( new_val > max_marg[t][j] ){
					max_marg[t][j] = new_val;
					argmax[t][j] = i;
				}
			}
		}
		for(int i=0;i<num_states;i++)
			max_marg[t][i] += nodePot[t*num_states+i];
	}
	//backward pass to find the ystar	
	y_decode->clear();
	y_decode->resize(num_vars);
	
	double max_val = -1e300;
	for(int i=0;i<num_states;i++){
		if( max_marg[num_vars-1][i] > max_val ){
			max_val = max_marg[num_vars-1][i];
			(*y_decode)[num_vars-1] = i;
		}
	}
	for(int t=num_vars-2;t>=0;t--){
		(*y_decode)[t] = argmax[t+1][ (*y_decode)[t+1] ];
	}
	
	for(int t=0;t<num_vars;t++)
		delete[] max_marg[t];
	delete[] max_marg;

	for(int t=0;t<num_vars;t++)
		delete[] argmax[t];
	delete[] argmax;
}


void chain_oracle(Param* param, Model* model, Instance* ins, Label* y, int ins_id, Label* ystar){
	
	int num_vars = ins->num_vars;
	int num_states = param->num_states;
	
	double* theta_unary = new double[num_vars*num_states];
	double* theta_pair = new double[num_states*num_states];
	
	for(int i=0;i<num_vars;i++){
		for(int j=0;j<num_states;j++){
			theta_unary[i*num_states + j] = kernel_inner_prod( param, model, &(ins->x_arr[i]), j, ins_id, i );
			//cerr << "(" << ins_id << "," << i << "," << j<< ")" << ", kv=" << theta_unary[i*num_states+j] << endl;
		}
	}
	for(int i=0;i<num_states*num_states;i++)
		theta_pair[i] = model->w_pair[i];
	
	//loss augmented
	if( y != NULL ){
		for(int i=0;i<num_vars;i++){
			for(int j=0;j<num_states;j++){
				if( j != y->at(i) )
					theta_unary[ i*num_states +j ] += 1.0/num_vars;
			}
		}
	}
	
	/*cerr << "theta_unary:" << endl;
	for(int i=0;i<num_vars;i++){
		for(int j=0;j<num_states;j++){
			cerr << theta_unary[i*num_states + j] << ", " ;
		}
		cerr << endl;
	}*/

	// decode
	chain_logDecode(theta_unary, theta_pair, num_vars, num_states, ystar);
	
	delete[] theta_unary;
	delete[] theta_pair;
}

#endif
