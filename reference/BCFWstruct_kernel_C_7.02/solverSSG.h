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

	int* L = new int[n];
	int* offset = new int[n];
	for(int i=0;i<n;i++){
		Instance* ins = data->at(i);
		L[i] = ins->num_vars;
	}
	offset[0] = 0;
	for(int i=1;i<n;i++)
		offset[i] = offset[i-1] + L[i-1];
	int numNodes = offset[n-1] + L[n-1];
	int numStates = param->num_states;
	int d = numStates*numStates; //number of features handled directly (linearly)
	cerr << "numNodes=" << numNodes << endl;
	
	//set up model
	Model* model = new Model();
	model->n = n;
	model->offset = offset;
	model->numNodes = numNodes;
	model->numStates = numStates;
	model->support_patterns = new vector<SparseVect*>();
	for(int i=0;i<n;i++)
		for(int j=0;j<L[i];j++)
			model->support_patterns->push_back( &(data->at(i)->x_arr[j]) );
	
	model->alpha = new SparseVect[numStates];
	model->w_pair = new double[d];
	for(int i=0;i<d;i++)
		model->w_pair[i] = 0.0;
	
	cerr << "allocate kernel cache space..." << endl;
	model->kernel_cache = new double*[numNodes];
	for(int i=0;i<numNodes;i++){
		model->kernel_cache[i] = new double[numNodes];
		for(int j=0;j<numNodes;j++)
			model->kernel_cache[i][j] = -1;
	}
	
	SparseVect* alpha_Avg = new SparseVect[numStates];
	double* w_pair_Avg = new double[d];
	if( options->do_weighted_average ){
		mat_assign(alpha_Avg, model->alpha, numStates);
		vassign(w_pair_Avg, model->w_pair, d);
	}
	
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
	double gamma, scalar; //step-size
	Label* ystar_i = new Label();
	SparseVect* phi_i = new SparseVect();
	SparseVect* phi_star = new SparseVect();
	SparseVect* alpha_s = new SparseVect[numStates];
	for(int p = 0; p < options->num_pass; p++ ){
		
		random_shuffle(index.begin(), index.end());
		for(int r=0;r<n_train;r++){
			
			int i = index[r];
			//cerr << "id=" << i+1 << endl;
			Instance* ins = data->at(i);
			Label* label = labels->at(i);
			
			//oracle
			maxOracle(param, model, ins, label, i, ystar_i);
			/*cerr <<"label:" ;
			for(int i=0;i<label->size();i++){
				cerr << label->at(i) << "," ;
			}
			cerr << endl;
			cerr <<"ystar:";
			for(int i=0;i<ystar_i->size();i++){
				cerr << ystar_i->at(i) << ",";
			}
			cerr << endl;*/
			
			//w_s, alpha_s
			scalar = 1.0/(lambda*n_train);
			phi(param, ins, label, phi_i);
			phi(param, ins, ystar_i, phi_star);
			
			clear(alpha_s, numStates);
			for(int j=0;j<label->size();j++){
				
				if( label->at(j) != ystar_i->at(j) ){
					alpha_s[ label->at(j) ].push_back(make_pair(offset[i]+j,scalar));
					alpha_s[ ystar_i->at(j) ].push_back(make_pair(offset[i]+j,-scalar));
				}
			}
			
			sumReduce(alpha_s);
			//update
			gamma = 1.0/k;
			//cerr << gamma << endl;
			vtimes( model->w_pair, d, (1.0-gamma) );
			vadd( model->w_pair, gamma*scalar*n_train, phi_i );
			vadd( model->w_pair, -gamma*scalar*n_train, phi_star );
			
			/*cerr << "alpha_s" << endl;
			print( cerr, alpha_s, numStates);
			cerr << "alpha" << endl;
			print( cerr, model->alpha, numStates);*/
			
			mat_add( (1.0-gamma), model->alpha, gamma*n_train, alpha_s, numStates, model->alpha );
			
			/*cerr << "combine" << endl;
			print( cerr, model->alpha, numStates);*/
			//cerr << nnz(model->alpha, numStates) << endl;
			//if( i+1 == 55 )
			//	exit(0);

			//averaging
			if( options->do_weighted_average ){
				double rho = 2.0 / (k+1) ;
				mat_add( (1-rho), alpha_Avg, rho, model->alpha, numStates, alpha_Avg );
				for(int i=0;i<d;i++){
					w_pair_Avg[i] = (1-rho)*w_pair_Avg[i] + rho*model->w_pair[i];
				}
			}

			if( k % 1000 == 0 )
				cerr << ".";
				//cerr << (k%n) << ",";
			k++;
		}
		
		minus_time -= omp_get_wtime();

		Model* model_debug = new Model(*model);
		model_debug->alpha = alpha_Avg;
		model_debug->w_pair = w_pair_Avg;

		double train_error = average_loss( param, 0, n_train, param->data, param->labels, maxOracle, model_debug );
		
		double testPred_time = -omp_get_wtime();
		double test_error = average_loss( param, n_train, n_train+n_test, param->data, param->labels, maxOracle, model_debug );
		testPred_time += omp_get_wtime();
		
		int nnz_alpha = 0;
		set<int> used_factor;
		for(int i=0;i<numStates;i++){
			SparseVect* sv = &(alpha_Avg[i]);
			for(SparseVect::iterator it=sv->begin();it!=sv->end();it++){
				used_factor.insert(it->first);
				nnz_alpha++;
			}
		}

		writeModel(param->modelFname, model_debug);

		minus_time += omp_get_wtime();
		double end = omp_get_wtime();
		cerr << endl << "#pass=" << p << ", train_err=" << train_error << ", test_err=" << test_error << ", time=" << (end-start-minus_time) << ", testPredTime=" << testPred_time << ", nnz(alpha)=" << nnz_alpha << ", #SV=" << used_factor.size() << endl;
		//cerr << endl << "#pass=" << p << ", train_err=" << train_error << endl;

		delete model_debug;
	}

	if( options->do_weighted_average ){
		mat_assign(model->alpha, alpha_Avg, numStates);
		vassign(model->w_pair, w_pair_Avg, d);
	}

	return model;
}
