#include "struct.h"
#include <iostream>
#include <algorithm>
#include <omp.h>

using namespace std;

Model* solverSSG(Param* param, Option* options){
	
	vector<Instance*>* data = param->data;
	vector<Label*>* labels = param->labels;
	double lambda = param->lambda;
	
	int n = data->size();
	int n_train = param->n_train;
	int n_test = param->n_test;
	int d = param->d;

	//set up model
	Model* model = new Model();
	model->d = d;
	
	model->w = new double[d];
	for(int i=0;i<d;i++)
		model->w[i] = 0.0;
	
	double* w_Avg = new double[d];
	if( options->do_weighted_average ){
		vassign(w_Avg, model->w, d);
	}
	Model* model_debug = new Model();
	//model_debug->w = model->w;
	model_debug->w = w_Avg;
	model_debug->d = d;
	
	cerr << "running SSG on "<< n_train << " examples..." << endl;

	void (*phi)(Param*,Instance*,Label*, SparseVect*) = param->featuremapFunc;
	void (*maxOracle)(Param*,Model*,Instance*,Label*,int, Label*) = param->oracleFunc;
	
	vector<int> index;
	for(int i=0;i<n_train;i++)
		index.push_back(i);
	//readVect("randList",index);
	
	double start = omp_get_wtime();
	double minus_time = 0.0;
	int k = 1;
	double gamma; //step-size
	Label* ystar_i = new Label();
	SparseVect* phi_i = new SparseVect();
	SparseVect* phi_star = new SparseVect();
	for(int p = 0; p < options->num_pass; p++ ){
		
		random_shuffle(index.begin(), index.end());
		for(int r=0;r<n_train;r++){
			
			int i = index[r];
			//cerr << "id=" << i+1 << endl;
			Instance* ins = data->at(i);
			Label* label = labels->at(i);
			
			//oracle
			maxOracle(param, model, ins, label, i,   ystar_i);
			/*for(Label::iterator it=ystar_i->begin(); it!=ystar_i->end(); it++)
				cerr << *it << " " ;
			cerr << endl;*/

			//w_s, alpha_s
			phi(param, ins, label,   phi_i);
			phi(param, ins, ystar_i,   phi_star);
			
			//update
			gamma = 1.0/k;
			//cerr << gamma << endl;
			vtimes( model->w, d,  (1.0-gamma) );
			vadd( model->w, gamma/lambda,  phi_i );
			vadd( model->w, -gamma/lambda, phi_star );
			
			//averaging
			if( options->do_weighted_average ){
				double rho = 2.0 / (k+1) ;
				for(int i=0;i<d;i++){
					w_Avg[i] = (1-rho)*w_Avg[i] + rho*model->w[i];
				}
			}
			
			if( k % n_test == 0 ){
				minus_time -= omp_get_wtime();

				double testPred_time = -omp_get_wtime();
				double test_error = average_loss( param, n_train, n_train+n_test, param->data, param->labels, maxOracle, model_debug );
				testPred_time += omp_get_wtime();
				double p_obj = primal_obj(param, 0, n_train, n_test, model_debug);

				minus_time += omp_get_wtime();
				double end = omp_get_wtime();
				cerr << endl << "#pass=" << p <<"-" << (k%n_train) << ", test_acc=" << (1-test_error) << ", time=" << (end-start-minus_time) << ", testPredTime=" << testPred_time << ", p-obj=" << p_obj <<  endl;
			}
			k++;
		}
		
	}

	if( options->do_weighted_average ){
		vassign(model->w, w_Avg, d);
	}
	
	return model;
}
