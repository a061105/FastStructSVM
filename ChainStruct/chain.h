#ifndef CHAIN_H
#define CHAIN_H
#include "util.h"

class Seq{
	public:
	vector<SparseVec*> features;
	vector<Int> labels;
       	Int T;
};

class ChainProblem{
	
	public:
	static map<string,Int> label_index_map;
	static vector<string> label_name_list;
	static Int D;
	static Int K;
	vector<Seq*> data;
	Int N;

	ChainProblem(char* fname){
		readData(fname);
	}
	
	void readData(char* fname){
		
		ifstream fin(fname);
		char* line = new char[LINE_LEN];
		
		Seq* seq = new Seq();
		Int d = -1;
		N = 0;
		while( !fin.eof() ){

			fin.getline(line, LINE_LEN);
			string line_str(line);
			
			if( line_str.length() < 2 && fin.eof() ){
				if(seq->labels.size()>0)
					data.push_back(seq);
				break;
			}else if( line_str.length() < 2 ){
				data.push_back(seq);
				seq = new Seq();
				continue;
			}
			vector<string> tokens = split(line_str, " ");
			//get label index
			Int lab_ind;
			map<string,Int>::iterator it;
			if(  (it=label_index_map.find(tokens[0])) == label_index_map.end() ){
				lab_ind = label_index_map.size();
				label_index_map.insert(make_pair(tokens[0],lab_ind));
			}else
				lab_ind = it->second;

			SparseVec* x = new SparseVec();
			for(Int i=1;i<tokens.size();i++){
				vector<string> kv = split(tokens[i],":");
				Int ind = atoi(kv[0].c_str());
				Float val = atof(kv[1].c_str());
				x->push_back(make_pair(ind,val));

				if( ind > d )
					d = ind;
			}
			seq->features.push_back(x);
			seq->labels.push_back(lab_ind);
			N++;
		}
		fin.close();
		
		d += 1; //bias
		if( D < d )
			D = d;

		for(Int i=0;i<data.size();i++)
			data[i]->T = data[i]->labels.size();

		label_name_list.resize(label_index_map.size());
		for(map<string,Int>::iterator it=label_index_map.begin();
				it!=label_index_map.end();
				it++)
			label_name_list[it->second] = it->first;
		
		K = label_index_map.size();

		delete[] line;
	}
};

map<string,Int> ChainProblem::label_index_map;
vector<string> ChainProblem::label_name_list;
Int ChainProblem::D = -1;
Int ChainProblem::K;

class Model{
	
	public:
	Model(Float** _w, Float** _v, ChainProblem* prob){
		D = prob->D;
		K = prob->K;
		label_name_list = &(prob->label_name_list);
		label_index_map = &(prob->label_index_map);
		w = _w;
		v = _v;
	}
	
	Float** w;
	Float** v;
	Int D;
	Int K;
	vector<string>* label_name_list;
	map<string,Int>* label_index_map;
};

class Param{

	public:
	char* trainFname;
	char* modelFname;
	Float C;
	ChainProblem* prob;
	ChainProblem* heldout_prob;

	int solver;
	int max_iter;
	Float eta; //Augmented-Lagrangian parameter
	bool using_brute_force;
	
	Param(){
		solver = 0;
		C = 10.0;
		max_iter =100000;
		eta = 0.1;
		heldout_prob = NULL;
		using_brute_force = false;
	}
};

#endif 
