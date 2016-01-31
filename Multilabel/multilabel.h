#ifndef MULTILABEL_H
#define MULTILABEL_H
#include "util.h"
#include "LPsparse.h"

class Instance{
	public:
	SparseVec feature;
	Labels labels;
};

class MultilabelProblem{
	
	public:
	static map<string,Int> label_index_map;
	static vector<string> label_name_list;
	static Int D;
	static Int K;
	vector<Instance*> data;
	Int N;
	
	MultilabelProblem(char* fname){
		readData(fname);
	}
	
	void readData(char* fname){
		
		ifstream fin(fname);
		char* line = new char[LINE_LEN];
		Int d = -1;
		while( !fin.eof() ){

			fin.getline(line, LINE_LEN);
			string line_str(line);

			if( line_str.length() < 2 && fin.eof() )
				break;
			
			size_t found = line_str.find("  ");
			while (found != string::npos){
				line_str = line_str.replace(found, 2, " ");
				found = line_str.find("  ");
			}
			found = line_str.find(", ");
			while (found != string::npos){
				line_str = line_str.replace(found, 2, ",");
				found = line_str.find(", ");
			}
			vector<string> tokens = split(line_str, " ");
			//get label index
			Instance* ins = new Instance();
			map<string,Int>::iterator it;
			Int st = 0;
			while (st < tokens.size() && tokens[st].find(":") == string::npos){
				// truncate , out
				if (tokens[st].size() == 0){
					st++;
					continue;
				}
				vector<string> subtokens = split(tokens[st], ",");
				for (vector<string>::iterator it_str = subtokens.begin(); it_str != subtokens.end(); it_str++){
					string str = *it_str;
					if( (it=label_index_map.find(str)) == label_index_map.end() ){
						ins->labels.push_back(label_index_map.size());
						label_index_map.insert(make_pair(str, ins->labels.back()));
					}else{
						ins->labels.push_back(it->second);
					}
				}
				st++;
			}

			for(Int i=st;i<tokens.size();i++){
				vector<string> kv = split(tokens[i],":");
				Int ind = atoi(kv[0].c_str());
				Float val = atof(kv[1].c_str());
				ins->feature.push_back(make_pair(ind,val));
				if( ind > d )
					d = ind;
			}
			
			data.push_back(ins);
		}
		fin.close();

		if (D < d+1){
			D = d+1; //adding bias
		}
		N = data.size();
		K = label_index_map.size();
		label_name_list.resize(K);
		for(map<string,Int>::iterator it=label_index_map.begin();
				it!=label_index_map.end();
				it++)
			label_name_list[it->second] = it->first;
		
		delete[] line;
	}
};

map<string,Int> MultilabelProblem::label_index_map;
vector<string> MultilabelProblem::label_name_list;
Int MultilabelProblem::D = -1;
Int MultilabelProblem::K;

class LP_Problem{
	public:
	int n;
	int nf;
	int m;
	int me;
	Constr* A; //m+me by n+nf
	double* b;
	Constr* At;//n+nf by m+me
	double* c;
	
	double* x;
	double* y;

	void solve(){
		LPsolve(n,nf,m,me, A,At,b,c,  x, y);
	}
};

class Model{
	
	public:
	Model(Float** _w, Float** _v, MultilabelProblem* prob){
		D = prob->D;
		K = prob->K;
		label_name_list = &(prob->label_name_list);
		label_index_map = &(prob->label_index_map);
		w = _w;
		v = _v;

		ins_pred_prob = NULL;
	}
	
	Float** w;
	Float** v;
	Int D;
	Int K;
	vector<string>* label_name_list;
	map<string,Int>* label_index_map;
	
	
	void LPpredict( Instance* ins, Int* pred ){
		
		if( ins_pred_prob == NULL )
			construct_LP();
		
		//set cost vector
		double* c = ins_pred_prob->c; // K+K*(K-1)/2*4 by 1
		//unigram 
		for(int i=0;i<K;i++)
			c[i] = 0.0;
		for(SparseVec::iterator it=ins->feature.begin(); it!=ins->feature.end(); it++){
			Float* wj = w[it->first];
			for(int i=0;i<K;i++)
				c[i] -= wj[i]*it->second;
		}
		//bigram
		int M = K*(K-1)/2;
		for(int i=K;i<K+M*4;i++)
			c[i] = 0.0;
		int ij4=0;
		for(int i=0;i<K;i++){
			for(int j=i+1;j<K;j++, ij4+=4){
				c[ K + ij4 + 3 ] = -v[i][j];
			}
		}
		
		//obtain primal solution x
		ins_pred_prob->solve();
		
		//Rounding
		double* x = ins_pred_prob->x;
		for(int i=0;i<K;i++){
			if( x[i] < 1e-2 ){
				pred[i] = 0;
				continue;
			}
			if( ((double)rand()/RAND_MAX) < x[i] )
				pred[i] = 1;
			else
				pred[i] = 0;
		}
	}

	void construct_LP(){
		
		ins_pred_prob = new LP_Problem();
		int M = K*(K-1)/2; //number of bigram factor
		int m = 0, nf=0, n=K+M*4, me=2*M + M; //consistency + simplex
		
		ins_pred_prob->m = m;
		ins_pred_prob->nf = nf;
		ins_pred_prob->n = n;
		ins_pred_prob->me = me;
		
		Constr* A = new Constr[m+me];
		double* b = new double[m+me];
		double* c = new double[n+nf];
		int row=0;
		//construct consistency constraint
		int ij4=0;
		for(int i=0;i<K;i++){ //for each bigram factor
			for(int j=i+1; j<K; j++, ij4+=4){
				//right consistency
				/*A[row].push_back(make_pair((K+ij4+2*0+1*0), 1.0));
				A[row].push_back(make_pair((K+ij4+2*1+1*0), 1.0));
				A[row].push_back(make_pair(j, 1.0)); //b00+b10=1-u1
				b[row++] = 1.0;
				*/
				A[row].push_back(make_pair((K+ij4+2*0+1*1), 1.0));
				A[row].push_back(make_pair((K+ij4+2*1+1*1), 1.0));
				A[row].push_back(make_pair(j, -1.0)); //b01+b11=u1
				b[row++] = 0.0;
				//left consistency
				/*A[row].push_back(make_pair((K+ij4+2*0+1*0), 1.0));
				A[row].push_back(make_pair((K+ij4+2*0+1*1), 1.0));
				A[row].push_back(make_pair(i, 1.0)); //b00+b01=1-u1
				b[row++] = 1.0;
				*/
				A[row].push_back(make_pair((K+ij4+2*1+1*0), 1.0));
				A[row].push_back(make_pair((K+ij4+2*1+1*1), 1.0));
				A[row].push_back(make_pair(i, -1.0));//b10+b11=u1
				b[row++] = 0.0;
			}
		}
		
		ij4=0;
		for(int i=0;i<K;i++){
			for(int j=i+1; j<K; j++, ij4+=4){
				
				for(int d=0;d<4;d++)
					A[row].push_back(make_pair((K+ij4+d), 1.0)); //simplex
				b[row++] = 1.0;
			}
		}
		
		assert( row == m+me );
		
		ConstrInv* At = new ConstrInv[n+nf];
		transpose(A, m+me, n+nf, At);
		ins_pred_prob->A = A;
		ins_pred_prob->b = b;
		ins_pred_prob->c = c;
		ins_pred_prob->At = At;

		ins_pred_prob->x = new double[n+nf];
		ins_pred_prob->y = new double[m+me];
	}

	private:
	LP_Problem* ins_pred_prob;
};

class Param{

	public:
	char* trainFname;
	char* heldoutFname;
	char* modelFname;
	Float C;
	MultilabelProblem* prob;
	MultilabelProblem* heldout_prob;

	Int solver;
	Int max_iter;
	Int early_terminate;
	Float admm_step_size;
	Float eta; //Augmented-Lagrangian parameter
	Int heldout_period;
	bool do_subSolve;
	Int split_up_rate;
	Int speed_up_rate;
	Param(){
		solver = 0;
		C = 1;
		max_iter = 1000;
		eta = 10;
		heldout_prob = NULL;
		heldoutFname = NULL;
		heldout_period = -1;
		early_terminate = -1;
		admm_step_size = 0.0;
		do_subSolve = true;
		split_up_rate = 1;
		speed_up_rate = -1;
	}
};



#endif 
