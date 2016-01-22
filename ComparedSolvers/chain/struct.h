#ifndef STRUCT_H
#define STRUCT_H

#include "util.h"
#include <vector>
#include <map>
#include <set>

using namespace std;

typedef vector<int> Label;
typedef vector<pair<int,double> > FracLabel;

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
		int d;
		double* w;
};

class Param{
	
	public:
	map<string,int> label_index_map;
	vector<string> label_name_list;
	void (*featuremapFunc)(Param*,Instance*, Label*, SparseVect*);
	void (*oracleFunc)(Param*,Model*,Instance*,Label*,int, Label*);
	double (*loss)(Param*,Label*,Label*);
	void (*readData)(char*,Param*);
	
	vector<Instance*>* data; //set by readData
	vector<Label*>* labels;
	int d;
	int num_states;
	int n_train; //set after readData
	int n_test;
	
	double lambda;   //set from cmdline
	char* modelFname;

	Param(){
		d = -1;
		data = new vector<Instance*>();
		labels = new vector<Label*>();
	}
};

class Option{
	
	public:
	int num_pass;
	bool do_weighted_average;
	
	Option(){
		num_pass = 50;
		do_weighted_average = 1;
	}
};


double average_loss( Param* param, int i_start, int i_end, vector<Instance*>* data, vector<Label*>* labels, void (*maxOracle)(Param*,Model*,Instance*,Label*,int, Label*), Model* model){

	double (*loss)(Param*,Label*,Label*) = param->loss;
	
	Label* ystar = new Label();
	double loss_term = 0.0;
	int num_pred_vars=0;
	for(int i=i_start;i<i_end;i++){

		maxOracle(param, model, data->at(i), NULL, i, ystar);
		loss_term += loss(param, labels->at(i), ystar);
		num_pred_vars += labels->at(i)->size();
	}
	
	delete ystar;
	
	return loss_term/num_pred_vars;
}

/*void writeModel(char* fname, Model* model){
	
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
*/
#endif
