#ifndef CHAIN_H
#define CHAIN_H
#include "util.h"

class Seq{
	public:
	vector<SparseVec*> features;
	vector<Int> labels; 
};

class ChainProblem{
	
	public:
	map<string,Int> label_index_map;
	vector<string> label_name_list;
	vector<Seq*> data;
	Int D;

	void readData(char* fname){
		
		ifstream fin(fname);
		char* line = new char[LINE_LEN];
		
		Seq* seq = new Seq();
		D = -1;
		while( !fin.eof() ){

			fin.getline(line, LINE_LEN);
			string line_str(line);
			
			if( line_str.length() < 2 && fin.eof() ){
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

				if( ind > D )
					D = ind;
			}
			seq->features.push_back(x);
			seq->labels.push_back(lab_ind);
		}
		fin.close();
		
		D += 1; //bias

		label_name_list.resize(label_index_map.size());
		for(map<string,Int>::iterator it=label_index_map.begin();
				it!=label_index_map.end();
				it++)
			label_name_list[it->second] = it->first;

		delete[] line;
	}
};

#endif 
