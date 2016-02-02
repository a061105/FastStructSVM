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

double primal_obj( Param* param, int i_start, int i_end, int n_sample,  Model* model){
	
	double (*loss)(Param*,Label*,Label*) = param->loss;
	vector<Instance*>* data = param->data;
	vector<Label*>* labels = param->labels;

	Label* ystar = new Label();
	SparseVect* phi_i = new SparseVect();
	SparseVect* phi_star = new SparseVect();
	SparseVect* psi_i = new SparseVect();

	int m = n_sample;
	if( m > i_end-i_start )
		m = i_end-i_start;
	double loss_term = 0.0;
	for(int r=0;r<m;r++){
		
		int i = rand()%(i_end-i_start) + i_start;

		Instance* ins = data->at(i);
		Label* label = labels->at(i);

		param->oracleFunc(param, model, ins, NULL, i, ystar);
		param->featuremapFunc(param, ins, label,   phi_i);
		param->featuremapFunc(param, ins, ystar,   phi_star);
		
		//loss_term += dot(phi_star, model->w) - dot(phi_i, model->w) + loss(param, labels->at(i), ystar)/labels->size();
		loss_term += dot(phi_star, model->w) - dot(phi_i, model->w) + loss(param, labels->at(i), ystar);
	}
	loss_term *= ((double)(i_end-i_start))/m;
	
	double reg_term = 0.0;
	double lambda_bar = param->lambda*param->n_train;
	for(int i=0; i<model->d; i++){
		double wbar_val = model->w[i] * lambda_bar;
		reg_term += wbar_val*wbar_val;
	}
	reg_term /= (2.0 * lambda_bar);
	
	delete ystar;
	delete phi_i;
	delete phi_star;
	delete psi_i;
	
	return reg_term + loss_term;
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
