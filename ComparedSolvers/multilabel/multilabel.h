#ifndef CHAIN_H
#define CHAIN_H
#include "struct.h"
#include "LPsparse.h"
#include <iostream>

/** Multilabel with K labels have K+K(K-1)/2 output variables per instance. 
 *  FracLabel is of maximum length K+K(K-1)/2
 */


double multilabel_loss(Param* param, Label* ytrue, FracLabel* ypred){
	
	Int K = param->num_states;
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
	for(Label::iterator it=ytrue->begin(); it!=ytrue->end(); it++)
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

void multilabel_featuremap(Param* param, Instance* ins, FracLabel* y, SparseVect* phi){
	
	Int K = param->num_states;
	Int bigram_offset = param->d - K*K; //dim = (D+1)K+K^2
	
	phi->clear();
	//unigram
	SparseVect* fea = &(ins->x_arr[0]);
	for(SparseVect::iterator it=fea->begin(); it!=fea->end(); it++){
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

class LP_Problem{
	public:
	Int n;
	Int nf;
	Int m;
	Int me;
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

LP_Problem* ins_pred_prob = NULL;

void construct_LP(Param* param){

	ins_pred_prob = new LP_Problem();
	
	Int K = param->num_states;
	
	Int M = K*(K-1)/2; //number of bigram factor
	Int m = 0, nf=0, n=K+M*4, me=2*M + M; //consistency + simplex
	
	ins_pred_prob->m = m;
	ins_pred_prob->nf = nf;
	ins_pred_prob->n = n;
	ins_pred_prob->me = me;
	
	Constr* A = new Constr[m+me];
	double* b = new double[m+me];
	double* c = new double[n+nf];
	Int row=0;
	//construct consistency constraInt
	Int ij4=0;
	for(Int i=0;i<K;i++){ //for each bigram factor
		for(Int j=i+1; j<K; j++, ij4+=4){
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
	for(Int i=0;i<K;i++){
		for(Int j=i+1; j<K; j++, ij4+=4){

			for(Int d=0;d<4;d++)
				A[row].push_back(make_pair((K+ij4+d), 1.0)); //simplex
			b[row++] = 1.0;
		}
	}

	ConstrInv* At = new ConstrInv[n+nf];
	transpose(A, m+me, n+nf, At);
	ins_pred_prob->A = A;
	ins_pred_prob->b = b;
	ins_pred_prob->c = c;
	ins_pred_prob->At = At;

	ins_pred_prob->x = new double[n+nf];
	ins_pred_prob->y = new double[m+me];
}


void multilabel_oracle(Param* param, Model* model, Instance* ins, Label* y, Int ins_id,    FracLabel* ystar){
	
	if( ins_pred_prob == NULL )
		construct_LP(param);
	
	//set cost vector
	Int K = param->num_states;
	double* c = ins_pred_prob->c; // K+K*(K-1)/2*4 by 1
	double* w = model->w;
	Int d = model->d;
	Int bigram_offset = d - K*K;
	/*if( y!=NULL){
		cerr << "ins=" << ins_id << ": ";
		for(Label::iterator it=y->begin(); it!=y->end(); it++)
			cerr << *it << ", ";
		cerr << endl;
	}*/
	//unigram 
	for(Int i=0;i<K;i++)
		c[i] = 0.0;
	for(SparseVect::iterator it=ins->x_arr[0].begin(); it!=ins->x_arr[0].end(); it++){
		for(Int i=0;i<K;i++)
			c[i] -= w[it->first*K+i]*it->second;
	}
	if( y != NULL ){ //loss-augmented decoding
		for(Int i=0;i<K;i++){
			if( find( y->begin(), y->end(), i ) == y->end() )
				c[i] -= 1.0;
			else
				c[i] -= -1.0;
		}
	}

	//bigram
	Int M = K*(K-1)/2;
	for(Int i=K;i<K+M*4;i++)
		c[i] = 0.0;
	Int ij4=0;
	for(Int i=0;i<K;i++){
		for(Int j=i+1;j<K;j++, ij4+=4){
			c[ K + ij4 + 3 ] = -w[bigram_offset + i*K + j];
		}
	}
	
	//obtain primal solution x
	ins_pred_prob->solve();
	
	//Rounding
	double* x = ins_pred_prob->x; //len=K + 4M, M=K(K-1)/2
	/*if( y==NULL ){
		for(Int i=0;i<ins_pred_prob->n;i++)
			cerr << x[i] << " " << c[i] << endl;
		cerr << "-----------------------------------" << endl;
	}*/
	
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

void multilabel_readData(char* dataFname, Param* param){

	map<string,Int>* label_index_map = &(param->label_index_map);
	vector<string>* label_name_list = &(param->label_name_list);
	vector<Label*>* labels = param->labels;
	vector<Instance*>* data = param->data;
	
	ifstream fin(dataFname);
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
		
		Instance* ins = new Instance();
		ins->x_arr = new SparseVect[1];
		Label* label = new Label();
		labels->push_back(label);
		
		//get label indexes
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
				if( (it=label_index_map->find(str)) == label_index_map->end() ){
					Int lab_ind = label_index_map->size();
					label->push_back(lab_ind);
					label_index_map->insert(make_pair(str,lab_ind));
				}else{
					label->push_back(it->second);
				}
			}
			st++;
		}
		
		for(Int i=st;i<tokens.size();i++){
			vector<string> kv = split(tokens[i],":");
			Int ind = atoi(kv[0].c_str());
			double val = atof(kv[1].c_str());
			ins->x_arr[0].push_back(make_pair(ind,val));
			if( ind > d )
				d = ind;
		}
		data->push_back(ins);
	}
	fin.close();
	d = d+1; //adding bias

	Int K = label_index_map->size();
	for(Int i=0;i<data->size();i++)
		data->at(i)->num_vars = K;
	
	param->num_states = K;
	param->d = max( param->d, d*K + K*K );
	label_name_list->resize(K);
	for(map<string,Int>::iterator it=label_index_map->begin();
			it!=label_index_map->end();
			it++)
		(*label_name_list)[it->second] = it->first;
	
	delete[] line;
}

#endif
