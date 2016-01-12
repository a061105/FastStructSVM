#ifndef CHAIN_H
#define CHAIN_H
#include "util.h"
#include <cassert>
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
	static Int* remap_indices;
	static Int* rev_remap_indices;
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
		
		for(map<string,Int>::iterator it=label_index_map.begin(); it!=label_index_map.end(); it++){
			label_name_list[it->second] = it->first;
		}
		
		K = label_index_map.size();
		
		delete[] line;
	}
	void label_random_remap(){
		srand(time(NULL));
		if (remap_indices == NULL || rev_remap_indices == NULL){
			remap_indices = new Int[K];
			for (Int k = 0; k < K; k++)
				remap_indices[k] = k;
			random_shuffle(remap_indices, remap_indices+K);
			rev_remap_indices = new Int[K];
			for (Int k = 0; k < K; k++)
				rev_remap_indices[remap_indices[k]] = k;
			label_index_map.clear();
			for (Int ind = 0; ind < K; ind++){
				label_index_map.insert(make_pair(label_name_list[ind], remap_indices[ind]));
			}
			label_name_list.clear();
			label_name_list.resize(label_index_map.size());
			for(map<string,Int>::iterator it=label_index_map.begin(); it!=label_index_map.end(); it++){
				label_name_list[it->second] = it->first;
			}
		}
		for (int i = 0; i < data.size(); i++){
			Seq* seq = data[i];
			for (int t = 0; t < seq->T; t++){
				Int yi = seq->labels[t];
				seq->labels[t] = remap_indices[yi];
			}
		}
	}

	void print_freq(){
		Int* freq = new Int[K];
		memset(freq, 0, sizeof(Int)*K);
		for (int i = 0; i < data.size(); i++){
			Seq* seq = data[i];
			for (int t = 0; t < seq->T; t++){
				Int yi = seq->labels[t];
				freq[yi]++;
			}
		}
		sort(freq, freq + K, greater<Int>());
		for (int k = 0; k < K; k++)
			cout << freq[k] << " ";
		cout << endl;
	}

	void simple_shuffle(){
		
		for (int i = 0; i < data.size(); i++){
			Seq* seq = data[i];
			for (int t = 0; t < seq->T; t++){
				Int yi = seq->labels[t];
				seq->labels[t] = (yi+13)%K;
			}
		}
	}
};

map<string,Int> ChainProblem::label_index_map;
vector<string> ChainProblem::label_name_list;
Int ChainProblem::D = -1;
Int ChainProblem::K;
Int* ChainProblem::remap_indices=NULL;
Int* ChainProblem::rev_remap_indices=NULL;

class Model{
	
	public:
	Model(char* fname){
		ifstream fin(fname);
		char* line = new char[LINE_LEN];
		//first line, get K
		fin.getline(line, LINE_LEN);
		string line_str(line);
		vector<string> tokens = split(line_str, "=");
		K = stoi(tokens[1]);
		
		//second line, get label_name_list
		fin.getline(line, LINE_LEN);
		line_str = string(line);
		tokens = split(line_str, " ");
		label_name_list = new vector<string>();
		label_index_map = new map<string, Int>();
		//token[0] is 'label', means nothing
		for (Int i = 1; i < tokens.size(); i++){
			label_name_list->push_back(tokens[i]);
			label_index_map->insert(make_pair(tokens[i], (i-1)));
		}
		
		//third line, get D
		fin.getline(line, LINE_LEN);
		line_str = string(line);
		tokens = split(line_str, "=");
		D = stoi(tokens[1]);
		
		//skip fourth line
		fin.getline(line, LINE_LEN);

		//next D lines: read w
		w = new Float*[D];
		for (Int j = 0; j < D; j++){
			w[j] = new Float[K];
			fin.getline(line, LINE_LEN);
			line_str = string(line);
			tokens = split(line_str, " ");
			Float* wj = w[j];
			for (vector<string>::iterator it = tokens.begin(); it != tokens.end(); it++){
				vector<string> k_w = split(*it, ":");
				Int k = stoi(k_w[0]);
				wj[k] = stof(k_w[1]);
			}
		}

		//skip next line
		fin.getline(line, LINE_LEN);

		//next K lines: read v
		v = new Float*[K];
		for (Int k1 = 0; k1 < K; k1++){
			v[k1] = new Float[K];
			fin.getline(line, LINE_LEN);
			line_str = string(line);
			tokens = split(line_str, " ");
			Float* v_k1 = v[k1];
			for (vector<string>::iterator it = tokens.begin(); it != tokens.end(); it++){
				vector<string> k2_v = split(*it, ":");
				Int k2 = stoi(k2_v[0]);
				v_k1[k2] = stof(k2_v[1]);
			}
		}
		sparse_v = new vector<pair<Int, Float>>[K];
		for (Int k1 = 0; k1 < K; k1++){
			sparse_v[k1].clear();
			for (Int k2 = 0; k2 < K; k2++){
				if (fabs(v[k1][k2]) > 1e-12){
					sparse_v[k1].push_back(make_pair(k2, v[k1][k2]));
				}
			}
		}
	}

	Model(Float** _w, Float** _v, ChainProblem* prob){
		D = prob->D;
		K = prob->K;
		label_name_list = &(prob->label_name_list);
		label_index_map = &(prob->label_index_map);
		w = _w;
		v = _v;
		sparse_v = new vector<pair<Int, Float>>[K];
		for (Int k1 = 0; k1 < K; k1++){
			sparse_v[k1].clear();
			for (Int k2 = 0; k2 < K; k2++){
				if (fabs(v[k1][k2]) > 1e-12){
					sparse_v[k1].push_back(make_pair(k2, v[k1][k2]));
				}
			}
		}
	}
	
	~Model(){
		for (Int k = 0; k < K; k++){
		       	sparse_v[k].clear();
		}
		delete[] sparse_v;
	}

	Float** w;
	Float** v;
	Int D;
	Int K;
	vector<string>* label_name_list;
	map<string,Int>* label_index_map;
	vector<pair<Int, Float>>* sparse_v;

	void writeModel( char* fname ){

		ofstream fout(fname);
		fout << "nr_class K=" << K << endl;
		fout << "label ";
		for(vector<string>::iterator it=label_name_list->begin();
				it!=label_name_list->end(); it++)
			fout << *it << " ";
		fout << endl;
		fout << "nr_feature D=" << D << endl;
		fout << "unigram w, format: D lines; line j contains (k, w) forall w[j][k] neq 0" << endl;
		for (int j = 0; j < D; j++){
			Float* wj = w[j];
			bool flag = false;
			for (int k = 0; k < K; k++){
				if (fabs(wj[k]) < 1e-12)
					continue;
				if (flag)
					fout << " ";
				else
					flag = true;
				fout << k << ":" << wj[k];
			}
			fout << endl;
		}

		fout << "bigram v, format: K lines; line k1 contains (k2, v) forall v[k1][k2] neq 0" << endl;
		for (int k1 = 0; k1 < K; k1++){
			Float* v_k1 = v[k1];
			bool flag = false;
			for (int k2 = 0; k2 < K; k2++){
				if (fabs(v_k1[k2]) < 1e-12)
					continue;
				if (flag)
					fout << " ";
				else 
					flag = true;
				fout << k2 << ":" << v_k1[k2];
			}
			fout << endl;
		}
		fout.close();
	}

	Float calcAcc_Viterbi(ChainProblem* prob){
		vector<Seq*>* data = &(prob->data);
		Int nSeq = data->size();
		Int N = 0;
		Int hit=0;
		for(Int n=0;n<nSeq;n++){
			
			Seq* seq = data->at(n);
			N += seq->T;
			//compute prediction
			Int* pred = new Int[seq->T];
			Float** max_sum = new Float*[seq->T];
			Int** argmax_sum = new Int*[seq->T];
			for(Int t=0; t<seq->T; t++){
				max_sum[t] = new Float[K];
				argmax_sum[t] = new Int[K];
				for(Int k=0;k<K;k++)
					max_sum[t][k] = -1e300;
			}
			////Viterbi t=0
			SparseVec* xi = seq->features[0];
			for(Int k=0;k<K;k++)
				max_sum[0][k] = 0.0;
			for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
				Float* wj = w[it->first];
				for(Int k=0;k<K;k++)
					max_sum[0][k] += wj[k]*it->second;
			}
			////Viterbi t=1...T-1
			for(Int t=1; t<seq->T; t++){
				//passing message from t-1 to t
				for(Int k1=0;k1<K;k1++){
					Float tmp = max_sum[t-1][k1];
					Float cand_val;
					/*for(Int k2=0;k2<K;k2++){
						 cand_val = tmp + v[k1][k2];
						 if( cand_val > max_sum[t][k2] ){
							max_sum[t][k2] = cand_val;
							argmax_sum[t][k2] = k1;
						 }
					}*/
					for (vector<pair<Int, Float>>::iterator it = sparse_v[k1].begin(); it != sparse_v[k1].end(); it++){
						Int k2 = it->first; 
						cand_val = tmp + it->second;
						if( cand_val > max_sum[t][k2] ){
							max_sum[t][k2] = cand_val;
							argmax_sum[t][k2] = k1;
						}	
					}
				}
				//adding unigram factor
				SparseVec* xi = seq->features[t];
				for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++)
					for(Int k2=0;k2<K;k2++)
						max_sum[t][k2] += w[it->first][k2] * it->second;
			}
			////Viterbi traceback
			pred[seq->T-1] = argmax( max_sum[seq->T-1], K );
			for(Int t=seq->T-1; t>=1; t--){
				pred[t-1] = argmax_sum[t][ pred[t] ];
			}
			
			//compute accuracy
			for(Int t=0;t<seq->T;t++){
				assert(label_name_list == &(prob->label_name_list));
				//if( label_name_list->at(pred[t]) == prob->label_name_list[seq->labels[t]] )
				if( pred[t] == seq->labels[t] )
					hit++;
			}
			
			for(Int t=0; t<seq->T; t++){
				delete[] max_sum[t];
				delete[] argmax_sum[t];
			}
			delete[] max_sum;
			delete[] argmax_sum;
			delete[] pred;
		}
		Float acc = (Float)hit/N;
		
		return acc;
	}
};

class Param{

	public:
	char* trainFname;
	char* heldoutFname;
	char* modelFname;
	Float C;
	ChainProblem* prob;
	ChainProblem* heldout_prob;
	
	int solver;
	int max_iter;
	Float eta; //Augmented-Lagrangian parameter
	bool using_brute_force;
	int split_up_rate;
	int write_model_period;
	int early_terminate;
	Float admm_step_size;
	Param(){
		solver = 0;
		C = 1.0;
		max_iter =100000;
		eta = 0.1;
		heldout_prob = NULL;
		using_brute_force = false;
		split_up_rate = 1;
		write_model_period = 0;
		early_terminate = 3;
		admm_step_size = 1.0;
	}
};

#endif 
