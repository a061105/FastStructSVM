#ifndef STRUCT_H
#define STRUCT_H

#include "util.h"
#include <vector>
#include <map>
#include <set>

using namespace std;

typedef vector<int> Label;

class Instance{
	public:
	int num_vars;
	SparseVect* x_arr;
	Instance(){
		num_vars = 0;
		x_arr = NULL;
	}
};

class Model{
	public:
		int n;
		//int* L;
		int* offset;
		int numNodes;
		int numStates;
		vector<SparseVect*>* support_patterns; //#nodes by d
		SparseVect* alpha; // #states by #nodes
		double* w_pair;
		
		double** kernel_cache;
};

class Param{
	
	public:
	map<string,int> label_index_map;
	vector<string> label_name_list;
	vector<Instance*>* data;
	vector<Label*>* labels;
	int n_train;
	int n_test;
	void (*featuremapFunc)(Param*,Instance*, Label*, SparseVect*);
	void (*oracleFunc)(Param*,Model*,Instance*,Label*,int, Label*);
	double (*loss)(Param*,Label*,Label*);
	int num_states;
	double gamma;
	double lambda;
	char kernel_type;
	char* modelFname;
};

class Option{
	
	public:
	int num_pass;
	bool do_weighted_average;
	vector<Instance*>* test_data;
	vector<Label*>* test_labels;
	
	Option(vector<Instance*>* _test_data, vector<Label*>* _test_labels){
		num_pass = 50;
		do_weighted_average = 1;
		test_data = _test_data;
		test_labels = _test_labels;
	}
};


double average_loss( Param* param, int i_start, int i_end, vector<Instance*>* data, vector<Label*>* labels, void (*maxOracle)(Param*,Model*,Instance*,Label*,int, Label*), Model* model){

	double (*loss)(Param*,Label*,Label*) = param->loss;
	
	Label* ystar = new Label();
	double loss_term = 0.0;
	for(int i=i_start;i<i_end;i++){

		maxOracle(param, model, data->at(i), NULL, i, ystar);
		loss_term += loss(param, labels->at(i), ystar);
	}
	loss_term /= (i_end-i_start);

	delete ystar;
	
	return loss_term;
}

void writeModel(char* fname, Model* model){
	
	ofstream fout(fname);
	int d =  model->numStates * model->numStates;
	fout << "bigram_factor:\t" << d << endl;
	for(int i=0;i<d;i++)
		fout << model->w_pair[i] << " ";
	fout << endl;
	
	set<int> used_factor;
	int nnz=0;
	for(int i=0;i<model->numStates;i++){
		for(SparseVect::iterator it=model->alpha[i].begin();
				it!=model->alpha[i].end();
				it++){
			
			used_factor.insert(it->first);
			nnz++;
		}
	}
	fout << "unigram_factor(y,factor_index,alpha):\t" << nnz << endl;
	
	for(int i=0;i<model->numStates;i++){
		for(SparseVect::iterator it=model->alpha[i].begin();
				it!=model->alpha[i].end();
				it++){
			
			fout << i << " " << it->first << " " << it->second << " ";
			fout << endl;
		}
	}
	
	fout << "Support_Vectors(factor_index,vector)\t" << used_factor.size() << endl;
	for(set<int>::iterator it=used_factor.begin(); it!=used_factor.end(); it++){
		
		SparseVect* sv = model->support_patterns->at(*it);
		fout << *it << " ";
		for(SparseVect::iterator it=sv->begin(); it!=sv->end(); it++)
			fout << it->first << ":" << it->second << " ";
		fout << endl;
	}
	
	fout.close();
}

#endif
