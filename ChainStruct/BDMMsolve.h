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
		admm_step_size = 1.0;

		//Compute unigram and bigram offset[i] = \sum_{j=1}^{i-1} T_j
		compute_offset();
		N = uni_offset[nSeq-1] + data->at(nSeq-1)->T; //#unigram factor
		M = bi_offset[nSeq-1] + data->at(nSeq-1)->T - 1; //#bigram factor
		cerr << "M=" << M << endl;
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
		messages = new Float*[2*M];
		for(Int i=0;i<2*M;i++){
			mu[i] = new Float[K];
			messages[i] = new Float[K];
			memset(mu[i], 0.0, sizeof(Float)*K);
			memset(messages[i], 0.0, sizeof(Float)*K);
		}
		
		//pre-allocate some algorithmic constants
		Q_diag = new Float[N];
		Int i=0;
		for(Int n=0;n<nSeq;n++){
			Seq* seq = data->at(n);
			for(Int t=0;t<seq->T;t++){
				Q_diag[i] = eta;
				for(SparseVec::iterator it=seq->features[t]->begin(); it!=seq->features[t]->end(); it++){
					Q_diag[i] += it->second * it->second;
				}
				i++;
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
	
	Model* solve(){
		
		Int* uni_ind = new Int[N];
		for(Int i=0;i<N;i++)
			uni_ind[i] = i;
		Int* bi_ind = new Int[M];
		for(Int i=0;i<M;i++)
			bi_ind[i] = i;
		
		//BDMM main loop
		Float* alpha_new = new Float[K];
		Float* beta_new = new Float[K*K];
		Float p_inf;
		for(Int iter=0;iter<max_iter;iter++){
			
			random_shuffle(uni_ind, uni_ind+N);
			//update unigram dual variables
			for(Int r=0;r<N;r++){
				Int i = uni_ind[r];
				Int n, t;
				get_uni_rev_index(i, n, t);
				
				//subproblem solving
				uni_subSolve(i, n, t, alpha_new);
				//maintain relationship between w and alpha
				Float* alpha_i = alpha[i];
				Seq* seq = data->at(n);
				SparseVec* xi = seq->features[t];
				for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
					Int j = it->first;
					Float fval = it->second;
					Float* wj = w[j];
					for(Int k=0;k<K;k++){
						wj[k] += fval * (alpha_new[k]-alpha_i[k]);
					}
				}
				//maintain messages(alpha) = (E*beta-alpha+\frac{1}{eta}mu)
				Float* msg_to_right = NULL;
				if( t != 0 ){
					Float* msg_to_left = messages[2*bi_index(n,t-1)+F_RIGHT];
					for(Int k=0;k<K;k++)
						msg_to_left[k] -= alpha_new[k] - alpha_i[k];
				}
				if( t != seq->T-1 ){
					Float* msg_to_right = messages[2*bi_index(n,t)+F_LEFT];
					for(Int k=0;k<K;k++)
						msg_to_right[k] -= alpha_new[k] + alpha_i[k];
				}
				
				//update alpha
				for(Int k=0;k<K;k++){
					alpha_i[k] = alpha_new[k];
				}
			}
			
			/*random_shuffle(bi_ind, bi_ind+M);
			//update bigram dual variables
			for(Int r=0;r<M;r++){
				Int i = bi_ind[r];
				Int n, t;
				get_bi_rev_index(i, n, t);
				
				//subproblem solving
				bi_subSolve(i, n, t, beta_new);
				
				//maintain relationship between v and beta
				Float* beta_i = beta[i];
				for(Int k=0;k<K;k++){
					Int Kk = K*k;
					for(Int k2=0;k2<K;k2++)
						v[k][k2] += beta_new[Kk+k2] - beta_i[Kk+k2];
				}
				//maintain messages(beta) = (E*beta-alpha+\frac{1}{eta}mu)
				Float* msg_to_left = messages[2*bi_index(n,t)+F_LEFT];
				Float* msg_to_right = messages[2*bi_index(n,t)+F_RIGHT];
				for(Int k=0; k<K; k++){
					Int Kk = K*k;
					for(Int k2=0; k2<K; k2++){
						msg_to_left[k] += beta_new[Kk+k2] - beta_i[Kk+k2];
						msg_to_right[k2] += beta_new[Kk+k2] - beta_i[Kk+k2];
					}
				}
				
				//update beta
				for(Int k=0;k<K;k++){
					Int Kk = K*k;
					for(Int k2=0;k2<K;k2++)
						beta_i[Kk+k2] = beta_new[Kk+k2];
				}
			}
			*/
			
			//ADMM update (enforcing consistency)
			Float mu_ijk;
			Float p_inf_ijk;
			p_inf = 0.0;
			for(Int n=0;n<nSeq;n++){
				Seq* seq = data->at(n);
				for(Int t=0;t<seq->T;t++){
					Int i2 = bi_index(n,t);
					cerr << "i2=" << i2 << endl;
					Int i1 = uni_index(n,t);
					cerr << "i1=" << i1 << ", 2M=" << 2*M << endl;
					for(Int j=0;j<1;j++){
						Float* mu_ij = mu[2*i2+j];
						Float* msg_ij = messages[2*i2+j];
						for(Int k=0;k<K;k++){
							mu_ijk = mu_ij[k];
							p_inf_ijk = -alpha[i1+j][k];
							//update
							mu_ij[k] += -admm_step_size*(p_inf_ijk);
							//maintain messages(mu) = (E*beta-alpha+\frac{1}{eta}mu)
							msg_ij[k] += mu_ij[k] - mu_ijk;
							//compute infeasibility of consistency constraint
							p_inf += fabs(p_inf_ijk);
						}
					}
				}
			}
			p_inf /= (2*M);
			
			cerr << "i=" << iter << ", infea=" << p_inf << ", Acc=" << train_acc_unigram() << endl;
		}
		
		delete[] uni_ind;
		delete[] bi_ind;
		delete[] alpha_new;
		delete[] beta_new;
		return new Model(w,v,prob);
	}

	private:
	
	void uni_subSolve(Int i, Int n, Int t, Float* alpha_new){ //solve i-th unigram factor
		
		Float* grad = new Float[K];
		Float* Dk = new Float[K];
		
		//data
		Seq* seq = data->at(n);
		Int yi = seq->labels[t];
		SparseVec* xi = seq->features[t];
		//variable values
		Float Qii = Q_diag[i];
		Float* alpha_i = alpha[i];
		Float* msg_from_left=NULL;
		Float* msg_from_right=NULL;
		if( t!=0 )//not beginning
			msg_from_left = messages[2*bi_index(n,t-1)+F_RIGHT];
		if( t!=seq->T-1 ) //not end
			msg_from_right = messages[2*bi_index(n,t)+F_LEFT];
		
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
				grad[k] += w[ f_ind ][k] * f_val ;
		}
		
		//message=(E\beta-\alpha+\mu/\eta)
		if( msg_from_left != NULL)
			for(Int k=0;k<K;k++)
				grad[k] -= eta*msg_from_left[k];
		
		if( msg_from_right != NULL )
			for(Int k=0;k<K;k++)
				grad[k] -= eta*msg_from_right[k];
		
		//compute Dk
		for(Int k=0;k<K;k++){
			if( k != yi )
				Dk[k] = grad[k];
			else
				Dk[k] = grad[k] + Qii*C;
		}

		//sort according to D_k
		sort( Dk, Dk+K, greater<Float>() );
		
		//compute b by traversing D_k in descending order
		Float bb = Dk[0] - Qii*C;
		Int r;
		for( r=1; r<K && bb<r*Dk[r]; r++)
			bb += Dk[r];
		bb = bb / r;
		
		//record alpha new values
		for(Int k=0;k<K;k++)
			alpha_new[k] = min( (Float)((k!=yi)?0.0:C), (bb-grad[k])/Qii );
		
		delete[] grad;
		delete[] Dk;
	}
	
	void bi_subSolve(Int i, Int n, Int t, Float* beta_new){
		
		Int Ksq = K*K;
		Float* grad = new Float[Ksq];
		Float* Dk = new Float[Ksq];
		
		//data
		Seq* seq = data->at(n);
		Int yi = seq->labels[t];
		Int yj = seq->labels[t+1];
		Int yi_yj = yi*K + yj;

		//variable values
		Float* beta_i = beta[i];
		Float* msg_from_left = messages[2*bi_index(n,t)+F_LEFT];
		Float* msg_from_right = messages[2*bi_index(n,t)+F_RIGHT];
		
		//compute gradient
		Float Qii = (1.0+eta);
		for(Int k1k2=0; k1k2<Ksq; k1k2++)
			grad[k1k2] = - Qii*beta_i[k1k2];
		
		for(Int k1=0;k1<K;k1++){
			Int Kk1 = K*k1;
			for(Int k2=0;k2<K;k2++)
				grad[Kk1+k2] += v[k1][k2];
		}
		//grad: message from left
		for(Int k1=0;k1<K;k1++){
			Int Kk1 = k1*K;
			Float tmp = eta*msg_from_left[k1];
			for(Int k2=0;k2<K;k2++)
				grad[Kk1+k2] += tmp;
		}
		//grad: message from right
		for(Int k1=0;k1<K;k1++){
			Int Kk1 = k1*K;
			for(Int k2=0;k2<K;k2++)
				grad[Kk1+k2] += eta*msg_from_right[k2];
		}
		
		//compute Dk
		for(Int k1k2=0;k1k2<Ksq;k1k2++){
			if( k1k2 != yi_yj )
				Dk[k1k2] = grad[k1k2];
			else
				Dk[k1k2] = grad[k1k2] + Qii*C;
		}
		
		//sort according to D_k
		sort( Dk, Dk+Ksq, greater<Float>() );
		
		//compute b by traversing D_k in descending order
		Float b = Dk[0] - Qii*C;
		Int r;
		for( r=1; r<Ksq && b<r*Dk[r]; r++)
			b += Dk[r];
		b = b / r;
		
		//record alpha new values
		for(Int k1k2=0;k1k2<Ksq;k1k2++)
			beta_new[k1k2] = min( (Float)((k1k2!=yi_yj)?0.0:C), (b-grad[k1k2])/Qii );
		
		delete[] grad;
		delete[] Dk;
	}
	
	void compute_offset(){
		
		uni_offset = new Int[nSeq];
		bi_offset = new Int[nSeq];
		uni_offset[0] = 0;
		bi_offset[0] = 0;
		for(Int i=1;i<nSeq;i++){
			uni_offset[i] = uni_offset[i-1] + data->at(i-1)->T;
			bi_offset[i] = bi_offset[i-1] + data->at(i-1)->T-1;
		}
	}

	inline Int uni_index(Int n, Int t){
		return uni_offset[n]+t;
	}
	inline Int bi_index(Int n, Int t){
		return bi_offset[n]+t;
	}
	inline void get_uni_rev_index(Int i, Int& n, Int& t){
		n=1;
		while( n<nSeq && i >= uni_offset[n] )n++;
		n -= 1;
		
		t = i-uni_offset[n];
	}
	inline void get_bi_rev_index(Int i, Int& n, Int& t){
		n=1;
		while( n<nSeq && i >= bi_offset[n] )n++;
		n -= 1;
		
		t = i - bi_offset[n];
	}
	
	Float train_acc_unigram(){
		
		Float* prod = new Float[K];
		Int hit=0;
		for(Int n=0;n<nSeq;n++){
			Seq* seq = data->at(n);
			for(Int t=0; t<seq->T; t++){
				
				SparseVec* xi = seq->features[t];
				Int yi = seq->labels[t];
				memset(prod, 0.0, sizeof(Float)*K);
				for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
					Float* wj = w[it->first];
					for(Int k=0;k<K;k++)
						prod[k] += wj[k]*it->second;
				}
				Float max_val = -1e300;
				Int argmax;
				for(Int k=0;k<K;k++)
					if( prod[k] > max_val ){
						max_val = prod[k];
						argmax = k;
					}
				
				if( argmax == yi )
					hit++;
			}
		}
		Float acc = (Float)hit/N;
		
		delete[] prod;
		return acc;
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
	Float** messages;// 2M*K message=(E*beta-alpha+\frac{1}{\eta}\mu)
	enum Direction {F_LEFT=0, F_RIGHT=1};
	
	Int max_iter;
	Float eta;
	Float admm_step_size;
};
