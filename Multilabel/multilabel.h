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

	public:
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
	bool using_brute_force_search;
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
		using_brute_force_search = false;
	}
};

void labelToFracLabel( Param* param, Labels* labels, FracLabel* y ){
	
	Int K = param->prob->K;
	y->clear();
	for(Labels::iterator it=labels->begin(); it!=labels->end(); it++)
		y->push_back(make_pair(*it,1.0));
	for(Labels::iterator it=labels->begin(); it!=labels->end(); it++)
		for(Labels::iterator it2=labels->begin(); it2!=labels->end(); it2++)
			if( *it2 > *it )
				y->push_back(make_pair( K + (*it)*K + (*it2), 1.0 ));
}

void multilabel_oracle(Param* param, Model* model, Instance* ins, Labels* y, FracLabel* ystar){
	
	LP_Problem* ins_pred_prob = model->ins_pred_prob;
	if( ins_pred_prob == NULL ){
		model->construct_LP();
		ins_pred_prob = model->ins_pred_prob;
	}
	
	//set cost vector
	Int K = param->prob->K;
	double* c = ins_pred_prob->c; // K+K*(K-1)/2*4 by 1
	Float** w = model->w;
	Float** v = model->v;
	
	//unigram 
	for(Int i=0;i<K;i++)
		c[i] = 0.0;
	for(SparseVec::iterator it=ins->feature.begin(); it!=ins->feature.end(); it++){
		Int j = it->first;
		Float* wj = w[j];
		Float xij = it->second;
		for(Int k=0; k<K; k++)
			c[k] -= w[j][k]*xij;
	}
	
	if( y != NULL ){ //loss-augmented decoding
		for(Int k=0; k<K; k++){
			if( find( y->begin(), y->end(), k ) == y->end() )
				c[k] -= 1.0;
			else
				c[k] -= -1.0;
		}
	}

	//bigram
	Int M = K*(K-1)/2;
	for(Int i=K;i<K+M*4;i++)
		c[i] = 0.0;
	Int ij4=0;
	for(Int i=0;i<K; i++){
		for(Int j=i+1; j<K; j++, ij4+=4){
			c[ K + ij4 + 3 ] = -v[i][j];
		}
	}
	
	//obtain primal solution x
	ins_pred_prob->solve();
	
	//Rounding
	double* x = ins_pred_prob->x; //len=K + 4M, M=K(K-1)/2
	
	ystar->clear();
	for(Int k=0;k<K;k++){
		double v = x[k];
	        if( v < 1e-3 )
			continue;
		if( v > 1-1e-3 )
			v = 1.0;

		ystar->push_back( make_pair(k,v) );
	}
	Int ij=0;
	for(Int i=0;i<K;i++){
		for(Int j=i+1; j<K; j++, ij++ ){
			double v = x[K+ij*4+3]; //taking beta_11
			if( v < 1e-3 )
				continue;
			if( v > 1-1e-3 )
				v = 1.0;
			ystar->push_back( make_pair(K + i*K+j, v) );
		}
	}
}

double multilabel_loss(Param* param, Labels* ytrue, FracLabel* ypred){
	
	Int K = param->prob->K;
	double* pred = new double[K]; // -1: incorrect, +1: correct
	Int* yi_arr = new Int[K]; // -1: incorrect, +1: correct
	
	//predict array
	for(Int i=0;i<K;i++)
		pred[i] = 0.0;
	for(FracLabel::iterator it=ypred->begin(); it!=ypred->end() && it->first < K; it++)
		pred[ it->first ] = it->second;
	
	//label array
	for(Int i=0;i<K;i++)
		yi_arr[i] = 0;
	for(Labels::iterator it=ytrue->begin(); it!=ytrue->end(); it++)
		yi_arr[*it] = 1;
	
	double h_loss = 0.0;
	for(Int i=0;i<K;i++){
		if( yi_arr[i] == 1 )
			h_loss += (1-pred[i]);
		else
			h_loss += pred[i];
	}
	
	delete[] pred;
	delete[] yi_arr;

	return h_loss;
}

void multilabel_featuremap(Param* param, Instance* ins, FracLabel* y, SparseVec* phi){ 
         
        Int K = param->prob->K; 
	Int D = param->prob->D;
        Int bigram_offset = D*K; //dim = (D+1)K+K^2 
         
        phi->clear();
        //unigram 
        SparseVec* fea = &(ins->feature);
        for(SparseVec::iterator it=fea->begin(); it!=fea->end(); it++){ 
                for(FracLabel::iterator it2=y->begin(); it2!=y->end() && it2->first < K ; it2++){ 
                         
                        phi->push_back( make_pair( it->first * K + it2->first, it->second*it2->second ) ); 
                } 
        } 
         
        //bigram 
        FracLabel::iterator it=y->begin(); 
        for(; it!=y->end() && it->first<K; it++); 
 
        for(; it!=y->end(); it++){ 
                phi->push_back(make_pair( bigram_offset+(it->first-K), it->second )); 
        } 
}

double dot(SparseVec* sv, Float** w, Float** v, Int D, Int K){

        double sum=0.0;
        for(SparseVec::iterator it=sv->begin(); it!=sv->end(); it++){
		Int offset = it->first;
		if (offset < D*K){
			Int j = offset / K, k = offset % K;
                	sum += w[j][k] * it->second;
		} else {
			offset -= D*K;
			Int k1 = offset / K, k2 = offset % K;
			sum += v[k1][k2] * it->second;
		}
        }
        return sum;
}

double primal_obj( Param* param, Int i_start, Int i_end,  Model* model){
	
	vector<Instance*>* data = &(param->prob->data);

	FracLabel* ystar = new FracLabel();
	FracLabel* yn = new FracLabel();
	SparseVec* phi_n = new SparseVec();
	SparseVec* phi_star = new SparseVec();
	double loss_term = 0.0;
	Float** w = model->w;
	Float** v = model->v;
	Int K = param->prob->K;
	Int D = param->prob->D;
	Int N = param->prob->N;

	for(Int n=i_start; n<i_end; n++){
		
		Instance* ins = data->at(n);
		Labels* labels = &(ins->labels);
		labelToFracLabel( param, labels, yn );

		multilabel_oracle(param, model, ins, NULL, ystar);
		multilabel_featuremap(param, ins, yn, phi_n);
		multilabel_featuremap(param, ins, ystar, phi_star);
		/*SparseVec* xt = &(ins->feature);
		for (SparseVec::iterator it_x = xt->begin(); it_x != xt->end(); it_x++){
			Int j = it_x->first;
			Int xt_j = it_x->second;
			for (FracLabel::iterator it_y = ystar->begin(); it_y != ystar->end() && it_y->first < K; it_y++){
				Int k = it_y->first;
				Float frac = it_y->second;
				loss_term += w[j][k] * xt_j * frac;
			}
			for (FracLabel::iterator it_y = yn->begin(); it_y != yn->end() && it_y->first < K; it_y++){
				Int k = it_y->first;
				Float frac = it_y->second;
				loss_term -= w[j][k] * xt_j * frac;
			}
		}

		for (FracLabel::iterator it_y = ystar->begin(); it_y != ystar->end(); it_y++){
			if (it_y->first < K) continue;
			Int k1k2 = it_y->first - K;
			Int k1 = k1k2 / K, k2 = k1k2 % K;
			assert(k1 < k2);
			Float frac = it_y->second;
			loss_term += frac * v[k1][k2];
		}
		
		for (FracLabel::iterator it_y = yn->begin(); it_y != yn->end(); it_y++){
			if (it_y->first < K) continue;
			Int k1k2 = it_y->first - K;
			Int k1 = k1k2 / K, k2 = k1k2 % K;
			assert(k1 < k2);
			Float frac = it_y->second;
			loss_term -= frac * v[k1][k2];
		}*/
		
		loss_term += dot(phi_star, w, v, D, K) - dot(phi_n, w, v, D, K) + multilabel_loss(param, labels, ystar);
	}
	
	double reg_term = 0.0;
	//lambda = 1.0/C
	for(int j = 0; j < D; j++){
		for (Int k = 0; k < K; k++){
			double wbar_val = w[j][k];
			reg_term += wbar_val*wbar_val;
		}
	}
	for(int k1 = 0; k1 < K; k1++){
		for (Int k2 = k1+1; k2 < K; k2++){
			double wbar_val = v[k1][k2];
			reg_term += wbar_val*wbar_val;
		}
	}
	reg_term /= (2*param->C);
	loss_term /= param->C;
	
	delete ystar;
	delete yn;
	
	return reg_term + loss_term;
}

#endif 
