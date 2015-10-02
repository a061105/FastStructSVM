#include "util.h"
#include "struct.h"
#include "chain.h"
#include "solverSSG.h"
#include "solverBCFW.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>

void readData(char* dataFname, char* tempFname, Param* param){
	
	ifstream fin(tempFname);
	if( fin.fail() ){
		cerr << "failed to open file '" << tempFname << "' " << endl;
		exit(0);
	}
	char* tmp_cstr = new char[LINE_LEN];

	//build label maps
	fin.getline(tmp_cstr, LINE_LEN); //filter one line
	fin.getline(tmp_cstr, LINE_LEN);
	string line(tmp_cstr);
	vector<string> tokens = split(line, " ");
	
	map<string,int>* label_map = &(param->label_index_map);
	vector<string>* list = &(param->label_name_list);
	for(int i=0;i<tokens.size();i++){
		label_map->insert(make_pair(tokens[i],label_map->size()));
		list->push_back(tokens[i]);
	}
	fin.close();

	//read chains
	vector<Instance*>* data = new vector<Instance*>();
	vector<Label*>* labels = new vector<Label*>();
	
	Instance* ins = new Instance();
	Label* label = new Label();
	fin.open(dataFname);
	if( fin.fail() ){
		cerr << "failed to open file '" << dataFname << "' " << endl;
		exit(0);
	}
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
		map<string,int>::iterator it = label_map->find(tokens[0]);
		if( it == label_map->end() ){
			cerr << "token '" << tokens[0] << "' not defined in template file" << endl;
			exit(0);		
		}
		label->push_back(it->second);
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
	
	fin.open(dataFname);
	vector<string> pair_vec;
	for(int i=0;i<data->size();i++){
		
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
			}
		}
		fin.getline(tmp_cstr,LINE_LEN);//filter one line of blank
		ins->x_arr = x_arr;
	}
	fin.close();
	
	param->data = data;
	param->labels = labels;

	delete[] tmp_cstr;
}

void readTestData(char* dataFname, Param* param){
	
	char* tmp_cstr = new char[LINE_LEN];
	
	map<string,int>* label_map = &(param->label_index_map);
	
	//read chains
	vector<Instance*>* data = param->data;
	vector<Label*>* labels = param->labels;
	int n_train = data->size();

	Instance* ins = new Instance();
	Label* label = new Label();
	ifstream fin(dataFname);
	if( fin.fail() ){
		cerr << "failed to open file '" << dataFname << "' " << endl;
		exit(0);
	}
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
		map<string,int>::iterator it = label_map->find(tokens[0]);
		if( it == label_map->end() ){
			cerr << "token '" << tokens[0] << "' not defined in template file" << endl;
			exit(0);		
		}
		label->push_back(it->second);
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
	int n_test = data->size()-n_train;
	
	fin.open(dataFname);
	vector<string> pair_vec;
	for(int i=n_train; i<n_train+n_test; i++){
		
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
			}
		}
		fin.getline(tmp_cstr,LINE_LEN);//filter one line of blank
		ins->x_arr = x_arr;
	}
	fin.close();

	delete[] tmp_cstr;
}


int main(int argc, char** argv){
	
	if( argc < 9 ){
		cerr << "structTrain [data] [template] [test_data] [lambda] [modelFname] [0:SSG/1:BCFW] [kernel='G'/'L'] [gamma] (max_iter)" << endl;
		exit(0);
	}

	char* dataFname = argv[1];
	char* tempFname = argv[2];
	char* testFname = argv[3];
	double lambda = atof(argv[4]);
	char* modelFname = argv[5];
	int method = atoi(argv[6]);
	char kernel_type = argv[7][0];
	double gamma = atof(argv[8]);
	int max_iter = 50;
       	if( argc > 9 )
		max_iter = atoi(argv[9]);
	
	Param* param = new Param();
	readData(dataFname, tempFname, param);
	param->n_train = param->data->size();
	readTestData(testFname, param);
	param->n_test = param->data->size() - param->n_train;
	
	param->featuremapFunc = chain_featuremap;
	param->oracleFunc = chain_oracle;
	param->loss = chain_loss;
	param->num_states = param->label_index_map.size();
	param->gamma = gamma;
	param->lambda = lambda/param->n_train;
	param->modelFname = modelFname;
	param->kernel_type = kernel_type;
	//param->lambda = 0.01;
	
	cerr << "#train_samples=" << param->n_train << endl;
	cerr << "#test_samples=" << param->n_test << endl;
	cerr << "|Y_f|=" << param->num_states << endl;
	cerr << "lambda=" << param->lambda << endl;

	Option* option = new Option(NULL,NULL);
	option->num_pass = max_iter;
	
	Model* model;
	if( method == 0 )
		model = solverSSG(param, option);
	else
		model = solverBCFW(param, option);

	//writeModel(modelFname, model);

	return 0;
}
