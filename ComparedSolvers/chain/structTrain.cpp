#include "util.h"
#include "struct.h"
#include "chain.h"
#include "solverSSG.h"
#include "solverBCFW.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>

int main(int argc, char** argv){
	
	if( argc < 1+5 ){
		cerr << "structTrain [data] [test_data] [lambda] [modelFname] [0:SSG/1:BCFW] (max_iter)" << endl;
		exit(0);
	}

	char* dataFname = argv[1];
	char* testFname = argv[2];
	double lambda = atof(argv[3]);
	char* modelFname = argv[4];
	int method = atoi(argv[5]);
	int max_iter = -1;
       	if( argc > 1+5 )
		max_iter = atoi(argv[6]);
	
	Param* param = new Param();
	param->featuremapFunc = chain_featuremap;
	param->oracleFunc = chain_oracle;
	param->loss = chain_loss;
	param->readData = chain_readData;
	
	param->readData(dataFname, param);
	param->n_train = param->data->size();
	param->readData(testFname, param);
	param->n_test = param->data->size() - param->n_train;
	
	param->lambda = lambda/param->n_train;
	param->modelFname = modelFname;
	//param->lambda = 0.01;
	
	cerr << "#train_samples=" << param->n_train << endl;
	cerr << "#test_samples=" << param->n_test << endl;
	cerr << "d=" << param->d << endl;
	cerr << "K=" << param->num_states << endl;
	cerr << "lambda=" << param->lambda << endl;

	Option* option = new Option();
	if( max_iter != -1 )
		option->num_pass = max_iter;
	
	Model* model;
	if( method == 0 )
		model = solverSSG(param, option);
	else
		model = solverBCFW(param, option);
	
	//writeModel(modelFname, model);

	return 0;
}
