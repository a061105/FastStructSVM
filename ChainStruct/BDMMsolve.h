#include "util.h"
#include "chain.h"

class BDMMsolve{
	
	public:
	BDMMsolve(Param* param){
		
		//Parse info from ChainProblem
		prob = param->prob;
		data = &(prob->data);
		nSeq = data->size();
		D = prob->D;
		K = prob->K;
		
		C = param->C;
		eta = param->eta;
		max_iter = param->max_iter;
		
		//Compute unigram and bigram offset[i] = \sum_{j=1}^{i-1} T_j
		compute_offset();
		N = uni_offset[nSeq-1] + data[nSeq-1]->T; //#unigram factor
		M = bi_offset[nSeq-1] + data[nSeq-1]->T-1; //#bigram factor
		//allocate dual variables
		alpha = new Float*[N];
		for(Int i=0;i<N;i++){
			alpha[i] = new Float[K];
			memset( alpha[i], 0.0, sizeof(Float)*K );
		}
		beta = new Float*[M];
		for(Int i=0;i<M;i++){
			beta[i] = new Float[K*K];
			memset( beta[i], 0.0, sizeof(Float)*K*K );
		}
		//allocate primal variables
		w = new Float*[D];
		for(Int j=0;j<D;j++){
			w[j] = new Float[K];
			memset( w[j], 0.0, sizeof(Float)*K );
		}
		v = new Float*[K];
		for(Int k=0;k<K;k++){
			v[k] = new Float[K];
			memset( v[k], 0.0, sizeof(Float)*K );
		}
		
		//allocating Lagrangian Multipliers for consistency constraints
		mu = new Float*[2*M]; //2 because of bigram
		msg = new Float*[2*M];
		for(Int i=0;i<2*M;i++){
			mu[i] = new Float[K];
			msg[i] = new Float[K];
			memset(mu[i], 0.0, sizeof(Float)*K);
			memset(msg[i], 0.0, sizeof(Float)*K);
		}
		
		//pre-allocate some algorithmic constants
		Q_diag = new Float[N];
		for(Int i=0;i<N;i++){
			Seq* seq = data[i];
			Q_diag[i] = 0.0;
			for(SparseVec::iterator it=seq->features.begin(); it!=seq->features.end(); it++){
				Q_diag[i] += it->second * it->second;
			}
		}
	}
	
	~BDMMsolve(){
		
		delete[] uni_offset;
		delete[] bi_offset;
		//dual variables
		for(Int i=0;i<N;i++)
			delete[] alpha[i];
		delete[] alpha;
		for(Int i=0;i<M;i++)
			delete[] beta[i];
		delete[] beta;
		//primal variables
		for(Int j=0;j<D;j++)
			delete[] w[j];
		delete[] w;
		for(Int k=0;k<K;k++)
			delete[] v[k];
		delete[] v;
		//Lagrangian Multiplier for Consistency constarints
		for(Int i=0;i<2*M;i++)
			delete[] mu[i];
		delete[] mu;
		//some constants
		delete Q_diag;
	}
	
	public Model* solve(){
		
		Int* uni_index = new Int[N];
		for(Int i=0;i<N;i++)
			uni_index[i] = i;
		Int* bi_index = new Int[M];
		for(Int i=0;i<M;i++)
			bi_index[i] = i;
		
		//BDMM main loop
		Float* alpha_new = new Float[K];
		Float* beta_new = new Float[K*K];
		for(Int iter=0;iter<max_iter;iter++){

			random_shuffle(uni_index, uni_index+N);
			//update unigram dual variables
			for(Int r=0;r<N;r++){
				Int i = uni_index[r];
				alpha_i = alpha[i];
				//subproblem solving
				uni_subSolve(i, alpha_new);
				//maintain relationship between w and alpha
				SparseVec* xi = data->at(i);
				for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
					Int j = it->first;
					Float fval = it->second;
					Float* wj = w[j];
					for(Int k=0;k<K;k++){
						wj[k] += fval * (alpha_new[k]-alpha_i[k]);
					}
				}
				//update alpha
				for(Int k=0;k<K;k++){
					alpha_i[k] = alpha_new[k];
				}
			}
			
			random_shuffle(bi_index, bi_index+M);
			//update bigram dual variables
			for(Int r=0;r<M;r++){
				Int i = bi_index[r];
				beta_i = beta[i];
				//subproblem solving
				bi_subSolve(i, beta_new);
				//maintain relationship between v and beta
				for(Int k=0;k<K;k++){
					Int Kk = K*k;
					for(Int k2=0;k2<K;k2++)
						v[k][k2] += beta_new[Kk+k2] - beta_i[Kk+k2];
				}
				//update beta
				for(Int k=0;k<K;k++){
					Int Kk = K*k;
					for(Int k2=0;k2<K;k2++)
						beta_i[Kk+k2] = beta_new[Kk+k2];
				}
			}
			
			//ADMM update (enforcing consistency)
		}
		
		delete[] uni_index;
		delete[] bi_index;
		return new Model();
	}

	private:
	
	void uni_subSolve(Int i, Float* alpha_new){ //solve i-th unigram factor
		
		Float* grad = new Float[K];
		Float* Dk = new Float[K];
		
		Int n, t;
		get_uni_rev_index(i, n, t);

		Seq* seq = data[n];
		Int yi = seq->labels[t];
		SparseVec* xi = seq->features[t];
		
		Float Qii = Q_diag[i];
		Float* alpha_i = alpha[i];

		for(Int k=0;k<K;k++){
			if( k!=yi )
				grad[k] = 1.0 - Qii*alpha_i[k];
			else
				grad[k] = -Qii*alpha_i[k];
		}

		//compute gradient (bottleneck is here)
		for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
			Int f_ind = it->first;
			Float f_val = it->second;
			for(Int k=0;k<K;k++)
				grad[k] += v[ f_ind ][k] * f_val ;
		}
		
		//compute Dk
		for(Int k=0;k<K;k++){
			if( k != yi )
				Dk[k] = grad[k];
			else
				Dk[k] = grad[k] + Qii*C;
		}

		//sort according to D_k
		
	}

	void bi_subSolve(){
		
	}

	void compute_offset(){
		
		uni_offset = new Int[nSeq];
		bi_offset = new Int[nSeq];
		uni_offset[0] = 0;
		bi_offset[0] = 0;
		for(Int i=1;i<nSeq;i++){
			uni_offset[i] = uni_offset[i-1] + data[i-1]->T;
			bi_offset[i] = bi_offset[i-1] + data[i-1]->T-1;
		}
	}

	inline Int uni_index(Int i, Int t){
		return uni_offset[i]+t;
	}
	inline Int bi_index(Int i, Int t){
		return bi_offset[i]+t;
	}
	inline void get_uni_rev_index(Int i, Int& n, Int& t){
		n=1;
		while( n<N && i >= uni_offset[n] )i++;
		n -= 1;
		
		t = i-uni_offset[n];
	}
	inline void get_bi_rev_index(Int i, Int& n, Int& t){
		n=1;
		while( n<M && i >= bi_offset[n] )i++;
		n -= 1;

		t = i - bi_offset[n];
	}
	
	
	ChainProblem* prob;
	
	vector<Seq*>* data;
	Float C;
	Int nSeq;
	Int N; //number of unigram factors (#variables)
	Int M; //number of bigram factors
	Int D;
	Int K;
	Int* uni_offset;
	Int* bi_offset;
	
	Float* Q_diag;
	Float** alpha; //N*K dual variables for unigram factor
	Float** beta; //M*K^2 dual variables for bigram factor
	
	Float** w; //D*K primal variables for unigram factor
	Float** v; //K^2 primal variables for bigram factor
	
	Float** mu; // 2M*K Lagrangian Multipliers on consistency constraInts
	Float** msg;// 2M*K message=(E*beta-alpha+\frac{1}{\eta}\mu)
	
	Int max_iter;
	Float eta;
};
