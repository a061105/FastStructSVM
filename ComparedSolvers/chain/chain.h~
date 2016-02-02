#ifndef CHAIN_H
#define CHAIN_H
#include "struct.h"
#include <iostream>

double chain_loss(Param* param, Label* ytrue, Label* ypred){ //assume of the same length
	
	int loss = 0;
	for(int i=0;i<ytrue->size();i++){
		if( ytrue->at(i) != ypred->at(i) )
			loss++;
	}
	return ((double)loss);
}

void chain_featuremap(Param* param, Instance* ins, Label* y, SparseVect* phi){
	
	int numVars = ins->num_vars;
	int numStates = param->num_states;
	int bigram_offset = param->d - numStates*numStates; //dim = (D+1)K+K^2
	
	phi->clear();
	//unigram
	for(int i=0;i<numVars;i++){
		SparseVect* fea = &(ins->x_arr[i]);
		for(SparseVect::iterator it=fea->begin(); it!=fea->end(); it++){
			phi->push_back( make_pair( it->first * numStates + y->at(i), it->second ) );
		}
	}
	
	//bigram
	for(int i=0;i<numVars-1;i++){
		int bi_ind = numStates * y->at(i) + y->at(i+1);
		phi->push_back(make_pair(bigram_offset+bi_ind,1.0));
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
		SparseVect* fea = &(ins->x_arr[i]);
		int i_offset = i*num_states;
		
		for(int j=0;j<num_states;j++)
			theta_unary[i_offset+j] = 0.0;
		for( SparseVect::iterator it=fea->begin(); it!=fea->end(); it++){
			int f_offset = it->first*num_states;
			double f_val = it->second;
			for(int j=0;j<num_states;j++){
				theta_unary[i_offset + j] += model->w[ f_offset + j] * f_val;
			}
		}
	}
	int bigram_offset = param->d - num_states*num_states;
	for(int i=0;i<num_states*num_states;i++)
		theta_pair[i] = model->w[bigram_offset + i];
	
	//loss augmented
	if( y != NULL ){
		for(int i=0;i<num_vars;i++){
			for(int j=0;j<num_states;j++){
				if( j != y->at(i) )
					theta_unary[ i*num_states +j ] += 1.0/num_vars;
					//theta_unary[ i*num_states +j ] += 1.0;
			}
		}
	}
	
	// decode
	chain_logDecode(theta_unary, theta_pair, num_vars, num_states, ystar);
	
	delete[] theta_unary;
	delete[] theta_pair;
}

void chain_readData(char* dataFname, Param* param){

	map<string,int>* label_index_map = &(param->label_index_map);
	vector<string>* label_name_list = &(param->label_name_list);
	
	vector<Instance*>* data = param->data;
	vector<Label*>* labels = param->labels;
	int data_old_size = data->size();

	//read chains
	
	Instance* ins = new Instance();
	Label* label = new Label();
	
	ifstream fin;
	fin.open(dataFname);
	if( fin.fail() ){
		cerr << "failed to open file '" << dataFname << "' " << endl;
		exit(0);
	}
	char* tmp_cstr = new char[LINE_LEN];
	vector<string> tokens;
	while( !fin.eof() ){
		
		fin.getline(tmp_cstr, LINE_LEN);
		string line(tmp_cstr);
		if( line.size() <= 1 && !fin.eof() ){
			data->push_back(ins);
			labels->push_back(label);
			ins = new Instance();
			label = new Label();
			continue;
		}else if( line.size() <= 1 ){
			break;
		}

		tokens = split(line," ");
		
		int label_ind;
		map<string,int>::iterator it = label_index_map->find(tokens[0]);
		if( it==label_index_map->end() ){
			label_ind = label_index_map->size();
			label_index_map->insert( make_pair(tokens[0],label_ind) );
		}else{
			label_ind = it->second;
		}
		
		label->push_back(label_ind);
		ins->num_vars++;
	}
	if( ins->num_vars != 0 ){
		data->push_back(ins);
		labels->push_back(label);
	}else{
		delete ins;
		delete label;
	}
	fin.close();
	
	//build label_index_map and label_name_list
	label_name_list->resize( label_index_map->size() );
	for(map<string,int>::iterator it=label_index_map->begin(); 
			it!= label_index_map->end(); it++){
		
		(*label_name_list)[it->second] = it->first;
	}
	param->num_states = label_name_list->size();
	
	fin.open(dataFname);
	vector<string> pair_vec;
	int max_ind = -1;
	for(int i=data_old_size;i<data->size();i++){
		
		Instance* ins = data->at(i);
		SparseVect* x_arr = new SparseVect[ins->num_vars];
		for(int j=0;j<ins->num_vars;j++){
			
			fin.getline(tmp_cstr,LINE_LEN);
			string line(tmp_cstr);
			tokens = split(line," ");
			
			for(int k=1;k<tokens.size();k++){
				pair_vec = split(tokens[k], ":");
				int ind = atoi( pair_vec[0].c_str() );
				double val = atof( pair_vec[1].c_str() );
				
				x_arr[j].push_back(make_pair(ind,val));

				if( ind > max_ind )
					max_ind = ind;
			}
		}
		fin.getline(tmp_cstr,LINE_LEN);//filter one line of blank
		ins->x_arr = x_arr;
	}
	fin.close();
	
	int K = param->num_states;
	int _d = (max_ind+1)*K + K*K;
	if( _d > param->d )
		param->d = _d;

	delete[] tmp_cstr;
}

#endif
