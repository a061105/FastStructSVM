#include "struct.h"
#include <iostream>
#include <algorithm>
#include <omp.h>

using namespace std;

double dual_obj( Param* param, double* w, double loss_term ){
	
	Int n_train = param->n_train;
	double lambda2 = param->lambda*n_train;
	double norm2_w = 0.0;
	for(Int j=0;j<param->d;j++){
		double w_bar = w[j]*lambda2;
		norm2_w += w_bar*w_bar;
	}

	return norm2_w/(2*lambda2) - loss_term*n_train;
}

Model* solverBCFW(Param* param, Option* options){
	
	vector<Instance*>* data = param->data;
	vector<Label*>* labels = param->labels;
	double lambda = param->lambda;
	
	Int n = data->size();
	Int n_train = param->n_train;
	Int n_test = param->n_test;
	Int d = param->d;
	//set up model
	Model* model = new Model();
	model->d = d;
	
	model->w = new double[d];
	for(Int i=0;i<d;i++)
		model->w[i] = 0.0;
	SparseVect* w_arr = new SparseVect[n_train];
	
	double* w_Avg = new double[d];
	if( options->do_weighted_average ){
		vassign(w_Avg, model->w, d);
	}
	Model* model_debug = new Model();
	//model_debug->w = model->w;
	model_debug->w = w_Avg;
	model_debug->d = d;
	
	//recording loss averaged by alpha_i
	double loss_term = 0.0;
	double* loss_arr = new double[n_train];
	for(Int i=0;i<n_train;i++)
		loss_arr[i] = 0.0;
	
	cerr << "running BCFW on "<< n_train << " examples..." << endl;

	void (*phi)(Param*,Instance*,FracLabel*, SparseVect*) = param->featuremapFunc;
	void (*maxOracle)(Param*,Model*,Instance*,Label*,Int, FracLabel*) = param->oracleFunc;
	
	vector<Int> index;
	for(Int i=0;i<n_train;i++)
		index.push_back(i);
	//readVect("randList",index);
	
	double start = omp_get_wtime();
	double minus_time = 0.0;
	Int k = 1;
	double gamma, scalar; //step-size
	FracLabel* ystar_i = new FracLabel();
	FracLabel* yi = new FracLabel();
	SparseVect* phi_i = new SparseVect();
	SparseVect* phi_star = new SparseVect();
	SparseVect* psi_i = new SparseVect();
	for(Int p = 0; p < options->num_pass; p++ ){
		
		random_shuffle(index.begin(), index.end());
		for(Int r=0;r<n_train;r++){
			
			Int i = index[r];
			//cerr << "id=" << i+1 << endl;
			Instance* ins = data->at(i);
			Label* label = labels->at(i);
			labelToFracLabel( param, label, yi );
			
			//oracle
			maxOracle(param, model, ins, label, i,   ystar_i);
			
			scalar = 1.0/(lambda*n_train);
			phi(param, ins, yi,   phi_i);
			phi(param, ins, ystar_i,   phi_star);
			sv_add( scalar, phi_i, -scalar, phi_star, psi_i);
			
			//compute line-search step
			double loss_s = multilabel_loss( param, label, ystar_i ) / (n_train);
			//gamma = (double)(2.0*n_train)/(k+2.0*n_train);
			double gamma_denom = 
				dot( w_arr+i, w_arr+i )+dot(psi_i,psi_i)-2.0*dot(w_arr+i, psi_i);
			double gamma_numer = 
				dot(w_arr+i,model->w)-dot(psi_i,model->w)-( loss_arr[i] - loss_s )/lambda;
			if( gamma_denom < 1e-5 )
				continue;

			double gamma = gamma_numer/gamma_denom;
			if( gamma > 1.0 ) gamma=1.0;
			else if( gamma < 0.0 ) gamma = 0.0;
			
			//maIntain w_i
			vadd( model->w, -1.0, &(w_arr[i]) );
			sv_add( (1-gamma), &(w_arr[i]), gamma, psi_i, &(w_arr[i]) );
			vadd( model->w, 1.0, &(w_arr[i]) );
			
			//maIntain loss term
			loss_term -= loss_arr[i];
			loss_arr[i] = (1-gamma)*loss_arr[i] + gamma*loss_s;
			loss_term += loss_arr[i];
			
			//averaging
			if( options->do_weighted_average ){
				double rho = 2.0 / (k+1) ;
				for(Int i=0;i<d;i++){
					w_Avg[i] = (1-rho)*w_Avg[i] + rho*model->w[i];
				}
			}
			
			if( k % n_test == 0 ){
			//if( k % 1 == 0 ){
				minus_time -= omp_get_wtime();

				double testPred_time = -omp_get_wtime();
				double test_error = average_loss( param, n_train, n_train + n_test, param->data, param->labels, maxOracle, model_debug);
				testPred_time += omp_get_wtime();
				
				double d_obj = dual_obj(param, model_debug->w, loss_term);
				double p_obj = primal_obj(param, 0, n_train, n_test, model_debug);
				
				minus_time += omp_get_wtime();
				double end = omp_get_wtime();
				cerr << endl << "#pass=" << p << "-" << (k%n_train) << ", test_acc=" << (1-test_error) << ", time=" << (end-start-minus_time) << ", testPredTime=" << testPred_time  << ", d-obj=" << d_obj << ", p-obj=" << p_obj << endl;
			}
			k++;
		}
		
	}
	
	if( options->do_weighted_average ){
		vassign(model->w, w_Avg, d);
	}
	
	delete[] loss_arr;
	delete[] w_Avg;	
        delete[] w_arr;
	delete[] model_debug;
	delete ystar_i;
	delete yi;
	delete phi_i;
	delete phi_star;
	delete psi_i;
	
	return model;
}
