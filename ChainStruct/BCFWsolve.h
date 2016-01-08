#include "util.h"
#include "chain.h"
#include <cassert>

extern long long line_bottom, line_top;
extern long long mat_bottom, mat_top;

class BCFWsolve{
	
	public:
	enum Direction {F_LEFT=0, F_RIGHT=1, NUM_DIRECT};
	
	BCFWsolve(Param* param){
		
		//Parse info from ChainProblem
		using_brute_force = param->using_brute_force;
		prob = param->prob;
		data = &(prob->data);
		nSeq = data->size();
		D = prob->D;
		K = prob->K;
		
		C = param->C;
		eta = param->eta;
		max_iter = param->max_iter;
		admm_step_size = 1.0;
		
		//bi_search
		inside = new bool[K*K];
		col_heap = new ArrayHeap[K];
		row_heap = new ArrayHeap[K];
		for (int k = 0; k < K; k++){
			row_heap[k] = new pair<Float, Int>[K];
			col_heap[k] = new pair<Float, Int>[K];	
		}	
		v_heap = new pair<Float, Int>[K*K];
		col_heap_size = new Int[K];
		row_heap_size = new Int[K];
		col_index = new Int[K*K];
		row_index = new Int[K*K];
		v_index = new Int[K*K];
		col_dir = new Int[K*K];
		row_dir = new Int[K*K];
		for (int kk = 0; kk < K*K; kk++){
			row_dir[kk] = kk / K;
			col_dir[kk] = kk % K;
		}
		
		//Compute unigram and bigram offset[i] = \sum_{j=1}^{i-1} T_j
		compute_offset();
		N = uni_offset[nSeq-1] + data->at(nSeq-1)->T; //#unigram factor
		M = bi_offset[nSeq-1] + data->at(nSeq-1)->T - 1; //#bigram factor
		//allocate dual variables
		alpha = new Float*[N];
		for(Int i=0;i<N;i++){
			alpha[i] = new Float[K];
			for(Int k=0;k<K;k++)
				alpha[i][k] = 0.0;
		}
		beta = new Float*[M];
		for(Int i=0;i<M;i++){
			beta[i] = new Float[K*K];
			for(Int kk=0;kk<K*K;kk++)
				beta[i][kk] = 0.0;
		}
		beta_suml = new Float*[M];
		for(Int i=0;i<M;i++){
			beta_suml[i] = new Float[K];
			for(Int k=0;k<K;k++)
				beta_suml[i][k] = 0.0;
		}
		beta_sumr = new Float*[M];
		for(Int i=0;i<M;i++){
			beta_sumr[i] = new Float[K];
			for(Int k=0;k<K;k++)
				beta_sumr[i][k] = 0.0;
		}
		
		//allocate primal variables
		w = new Float*[D];
		for(Int j=0;j<D;j++){
			w[j] = new Float[K];
			for(Int k=0;k<K;k++)
				w[j][k] = 0.0;
		}
		v = new Float*[K];
		for(Int k=0;k<K;k++){
			v[k] = new Float[K];
			for(Int k2=0;k2<K;k2++){
				v[k][k2] = 0.0;
				//update_v_heap
				v_heap[k*K+k2] = make_pair(0.0, k*K+k2);
				v_index[k*K+k2] = k*K+k2;
				//update_row_heap[k]
				row_heap[k][k2] = make_pair(0.0, k*K+k2);
				row_index[k*K+k2] = k2;
				//update_col_heap[k2]
				col_heap[k2][k] = make_pair(0.0, k*K+k2);
				col_index[k*K+k2] = k;
			}
			row_heap_size[k] = K;
			col_heap_size[k] = K;
		}
		v_heap_size = K*K;
		
		//allocating Lagrangian Multipliers for consistency constraInts
		mu = new Float*[2*M]; //2 because of bigram
		//messages = new Float*[2*M];
		for(Int i=0;i<2*M;i++){
			mu[i] = new Float[K];
			//messages[i] = new Float[K];
			for(Int k=0;k<K;k++)
				mu[i][k] = 0.0;
			//for(Int k=0;k<K;k++)
			//	messages[i][k] = 0.0;
		}
		
		//pre-allocate some algorithmic constants
		Q_diag = new Float[N];
		for(Int n=0;n<nSeq;n++){
			Seq* seq = data->at(n);
			for(Int t=0;t<seq->T;t++){
				Int i = uni_index( n, t );
				Q_diag[i] = eta;
				for(SparseVec::iterator it=seq->features[t]->begin(); it!=seq->features[t]->end(); it++){
					Q_diag[i] += it->second * it->second;
				}
			}
		}

		//uni_search
		max_indices = new Int[K];
		prod = new Float[K];
		
	}
	
	~BCFWsolve(){
		
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
		//Lagrangian Multiplier for Consistency constarInts
		for(Int i=0;i<2*M;i++){
			delete[] mu[i];
			//delete[] messages[i];
		}
		delete[] mu;
		//delete[] messages;
		
		//some constants
		delete Q_diag;

		//uni_search
		delete[] max_indices;
		delete[] prod;

		//bi_search
		delete[] inside;
		for (int k = 0; k < K; k++){
			delete[] col_heap[k];
			delete[] row_heap[k];
		}
		delete[] col_heap;
		delete[] row_heap;
		delete[] v_heap;
		delete[] col_heap_size;
		delete[] row_heap_size;
		delete[] col_index;
		delete[] row_index;
		delete[] v_index;
		delete[] col_dir;
		delete[] row_dir;
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
		Float* marg_ij = new Float[K];
		Float p_inf;

		//BCFW
		vector<int>* act_k_index = new vector<Int>[N];
		for (Int i = 0; i < N; i++){
			Int n,t;
			get_uni_rev_index(i, n, t);
			Seq* seq = data->at(n);
			act_k_index[i].push_back(seq->labels[t]);
		}
		vector<int>* act_kk_index = new vector<Int>[M];
		for (Int i = 0; i < M; i++){
			Int n,t;
			get_bi_rev_index(i, n, t);
			Seq* seq = data->at(n);
			act_kk_index[i].push_back(seq->labels[t]*K + seq->labels[t+1]);
		}

		for(Int iter=0;iter<max_iter;iter++){
			
			double bi_search_time = 0.0, bi_subSolve_time = 0.0, bi_maintain_time = 0.0;
			double uni_search_time = 0.0, uni_subSolve_time = 0.0, uni_maintain_time = 0.0;
			line_top = 0; line_bottom = 0;
			mat_top = 0;  mat_bottom = 0;
			random_shuffle(uni_ind, uni_ind+N);
			//update unigram dual variables
			for(Int r=0;r<N;r++){

				Int i = uni_ind[r];
				Int n, t;
				get_uni_rev_index(i, n, t);
			
				//brute force search
				uni_search_time -= omp_get_wtime();
				uni_search(i, n, t, act_k_index[i]);
				uni_search_time += omp_get_wtime();
				
				//subproblem solving
				uni_subSolve_time -= omp_get_wtime();
				uni_subSolve(i, n, t, act_k_index[i], alpha_new);
				uni_subSolve_time += omp_get_wtime();

				//maIntain relationship between w and alpha
				uni_maintain_time -= omp_get_wtime();
				Float* alpha_i = alpha[i];
				Seq* seq = data->at(n);
				SparseVec* xi = seq->features[t];
				Int yi = seq->labels[t];
				for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
					Int j = it->first;
					Float fval = it->second;
					Float* wj = w[j];
					for(vector<Int>::iterator it2 = act_k_index[i].begin(); it2 != act_k_index[i].end(); it2++){
						Int k = *it2;
						wj[k] += fval * (alpha_new[k]-alpha_i[k]);
					}
				}
				//maintain messages(alpha) = (E*beta-alpha+\frac{1}{eta}mu)
				/*if( t != 0 ){
					Float* msg_to_left = messages[2*bi_index(n,t-1)+F_RIGHT];
					for(Int k=0;k<K;k++)
						msg_to_left[k] -= alpha_new[k] - alpha_i[k];
				}
				if( t != seq->T-1 ){
					Float* msg_to_right = messages[2*bi_index(n,t)+F_LEFT];
					for(Int k=0;k<K;k++)
						msg_to_right[k] -= alpha_new[k] + alpha_i[k];
				}
				*/

				bool has_zero = 0;
				//update alpha
				for(vector<Int>::iterator it = act_k_index[i].begin(); it != act_k_index[i].end(); it++ ){
					Int k = *it;
					alpha_i[k] = alpha_new[k];	
					has_zero |= (fabs(alpha_new[k])<1e-12);
				}
					
				if (has_zero){
					vector<Int> tmp_vec;
					tmp_vec.reserve(act_k_index[i].size());
					for (vector<Int>::iterator it = act_k_index[i].begin(); it != act_k_index[i].end(); it++){
						Int k = *it;
						if ( fabs(alpha_i[k]) > 1e-12 || k == yi){
							tmp_vec.push_back(k);	
						}
					}
					act_k_index[i].clear();
					act_k_index[i] = tmp_vec;
				}
				uni_maintain_time += omp_get_wtime();
			}
			
			//update bigram dual variables
			random_shuffle(bi_ind, bi_ind+M);
			for(Int r=0;r<M;r++){
				Int i = bi_ind[r];
				Int n, t;
				get_bi_rev_index(i, n, t);
				Seq* seq = data->at(n);
				Int yi_l = seq->labels[t];
				Int yi_r = seq->labels[t+1];
				Int ylyr = yi_l*K + yi_r;
				
				//search using oracle
				bi_search_time -= omp_get_wtime();
				if (using_brute_force)
					bi_brute_force_search(i, n, t, act_kk_index[i]);	
				else
					bi_search(i, n, t, act_kk_index[i]);
				bi_search_time += omp_get_wtime();
				
				//subproblem solving
				bi_subSolve_time -= omp_get_wtime();
				bi_subSolve(i, n, t, act_kk_index[i], beta_new);
				bi_subSolve_time += omp_get_wtime();

				//maIntain relationship between v and beta
				bi_maintain_time -= omp_get_wtime();
				Float* beta_i = beta[i];
				for(vector<Int>::iterator it = act_kk_index[i].begin(); it != act_kk_index[i].end(); it++){
					Int k1k2 = *it;
					Int k2 = k1k2 % K;
					Int k1 = (k1k2 - k2)/K;
					Float delta_beta = beta_new[k1k2] - beta_i[k1k2];
					v[k1][k2] += delta_beta;
					//maintain v_heap
					Int index_k1k2 = v_index[k1k2];
					v_heap[index_k1k2].first += delta_beta;
					
					//maintain row_heap
					Int row_index_k1k2 = row_index[k1k2];
					row_heap[k1][row_index_k1k2].first += delta_beta;
					
					//maintain col_heap
					Int col_index_k1k2 = col_index[k1k2];
					col_heap[k2][col_index_k1k2].first += delta_beta;
					
					if (delta_beta >= 0.0){
						siftUp(v_heap, index_k1k2, v_index);
						siftUp(row_heap[k1], row_index_k1k2, row_index);
						siftUp(col_heap[k2], col_index_k1k2, col_index);
					} else {
						siftDown(v_heap, index_k1k2, v_index, v_heap_size);
						siftDown(row_heap[k1], row_index_k1k2, row_index, row_heap_size[k1]);
						siftDown(col_heap[k2], col_index_k1k2, col_index, col_heap_size[k2]);
					}
				}
				//maintain messages(beta) = (E*beta-alpha+\frac{1}{eta}mu)
				/*Float* msg_to_left = messages[2*bi_index(n,t)+F_LEFT];
				Float* msg_to_right = messages[2*bi_index(n,t)+F_RIGHT];
				for(Int k=0; k<K; k++){
					Int Kk = K*k;
					for(Int k2=0; k2<K; k2++){
						msg_to_left[k] += beta_new[Kk+k2] - beta_i[Kk+k2];
						msg_to_right[k2] += beta_new[Kk+k2] - beta_i[Kk+k2];
					}
				}
				*/

				//update beta and shrink active set if necessary
				bool has_zero = false;
				for(vector<Int>::iterator it = act_kk_index[i].begin(); it != act_kk_index[i].end(); it++){
					Int k1k2 = *it;
					Int k2 = k1k2 % K;
					Int k1 = (k1k2-k2)/K;
					beta_suml[i][k1] += beta_new[k1k2] - beta_i[k1k2];
					beta_sumr[i][k2] += beta_new[k1k2] - beta_i[k1k2];
					beta_i[k1k2] = beta_new[k1k2];
					has_zero |= (fabs(beta_new[k1k2]) < 1e-12);
				}
				if (has_zero){
					vector<Int> tmp_vec;
					tmp_vec.reserve(act_kk_index[i].size());
					for (vector<Int>::iterator it = act_kk_index[i].begin(); it != act_kk_index[i].end(); it++){
						Int k1k2 = *it;
						if ( fabs(beta_i[k1k2]) > 1e-12 || k1k2 == ylyr){
							tmp_vec.push_back(k1k2);
						}
					}
					act_kk_index[i].clear();
					act_kk_index[i] = tmp_vec;
				}
				bi_maintain_time += omp_get_wtime();
			}
			
			
			//ADMM update (enforcing consistency)
			bi_maintain_time -= omp_get_wtime();
			Float mu_ijk;
			Float p_inf_ijk;
			p_inf = 0.0;
			for(Int n=0;n<nSeq;n++){
				Seq* seq = data->at(n);
				for(Int t=0;t<seq->T-1;t++){
					Int i2 = bi_index(n,t);
					Int i1 = uni_index(n,t);
					for(Int j=0;j<NUM_DIRECT;j++){
						Float* mu_ij = mu[2*i2+j];
						//Float* msg_ij = messages[2*i2+j];
						//marginalize(beta[i2], (Direction)j, marg_ij);
						for(Int k=0;k<K;k++){
							//mu_ijk = mu_ij[k];
							//p_inf_ijk = marg_ij[k] - alpha[i1+j][k];
							if (j == 0)
								p_inf_ijk = beta_suml[i2][k] - alpha[i1][k];
							else
								p_inf_ijk = beta_sumr[i2][k] - alpha[i1+1][k];
							p_inf += fabs(p_inf_ijk);
							
							//p_inf_ijk = msg_ij[k] - mu_ijk;
							//update
							mu_ij[k] += admm_step_size*(p_inf_ijk);
							//maintain messages(mu) = (E*beta-alpha+\frac{1}{eta}mu)
							//msg_ij[k] += mu_ij[k] - mu_ijk;
							//compute infeasibility of consistency constraInt
						}
					}
				}
			}
			bi_maintain_time += omp_get_wtime();
			p_inf /= (2*M*K);
			
			Float nnz_alpha=0;
			for(Int i=0;i<N;i++){
				nnz_alpha += act_k_index[i].size();
			}
			nnz_alpha /= N;
			
			Float nnz_beta=0;
			for(Int i=0;i<M;i++){
				nnz_beta += act_kk_index[i].size();
			}
			nnz_beta /= M;
			
			double pos_rate = pos_count / (pos_count+neg_count+zero_count);
			double nz_rate = (pos_count+neg_count) / (pos_count+neg_count+zero_count);
			cerr << "i=" << iter << ", infea=" << p_inf << ", Acc=" << train_acc_Viterbi();
			cerr << ", nnz_a=" << nnz_alpha << ", nnz_b=" << nnz_beta ;
			cerr << ", uni_search=" << uni_search_time << ", uni_subSolve=" << uni_subSolve_time << ", uni_maintain=" << uni_maintain_time;
			cerr << ", bi_search="  << bi_search_time  << ", bi_subSolve="  << bi_subSolve_time  << ", bi_maintain="  << bi_maintain_time;
			cerr << ", p_rate=" << pos_rate << ", nz_rate=" << nz_rate ;
			cerr << ", avg area23=" << (double)line_top/line_bottom << ", avg area4=" << (double)mat_top/mat_bottom;
			cerr << endl;
			//if( p_inf < 1e-4 )
			//	break;
			
			//cerr << "i=" << iter << ", Acc=" << train_acc_Viterbi() << ", dual_obj=" << dual_obj() << endl;
		}
		
		delete[] marg_ij;
		delete[] uni_ind;
		delete[] bi_ind;
		delete[] alpha_new;
		delete[] beta_new;
		
		//search
		for (int i = 0; i < N; i++){
			act_k_index[i].clear();
		}
		for (int i = 0; i < M; i++){
			act_kk_index[i].clear();
		}
		delete[] act_k_index;
		delete[] act_kk_index;
		
		return new Model(w,v,prob);
	}

	private:
	
	void uni_subSolve(Int i, Int n, Int t, vector<Int>& act_uni_index, Float* alpha_new ){ //solve i-th unigram factor
		memset(prod, 0.0, sizeof(Float)*K);	
		Float* Dk = new Float[act_uni_index.size()];
		
		//data
		Seq* seq = data->at(n);
		Int yi = seq->labels[t];
		SparseVec* xi = seq->features[t];
		//variable values
		Float Qii = Q_diag[i];
		Float* alpha_i = alpha[i];
		Float* msg_from_left=NULL;
		Float* msg_from_right=NULL;
		/*if( t!=0 )//not beginning
			msg_from_left = messages[2*bi_index(n,t-1)+F_RIGHT];
		if( t!=seq->T-1 ) //not end
			msg_from_right = messages[2*bi_index(n,t)+F_LEFT];
		*/
		if( t != 0 ){
			Int i2 = bi_index(n,t-1);
			msg_from_left = new Float[K];
			Float* beta_suml_i2 = beta_suml[i2];
			for(vector<Int>::iterator it = act_uni_index.begin(); it != act_uni_index.end(); it++){
				Int k = *it;
				msg_from_left[k] = beta_suml_i2[k] -alpha[i][k] + mu[2*i2+F_RIGHT][k];
			}
		}
		if( t!=seq->T-1 ){
			Int i2 = bi_index(n,t);
			msg_from_right = new Float[K];
			Float* beta_sumr_i2 = beta_sumr[i2];
			for(vector<Int>::iterator it = act_uni_index.begin(); it != act_uni_index.end(); it++){
				Int k = *it;
				msg_from_right[k] = beta_sumr_i2[k] -alpha[i][k] + mu[2*i2+F_LEFT][k];
			}
		}
		
		for(vector<Int>::iterator it = act_uni_index.begin(); it != act_uni_index.end(); it++){
			Int k = *it;
			if( k!=yi )
				prod[k] = 1.0 - Qii*alpha_i[k];
			else
				prod[k] = -Qii*alpha_i[k];
		}
		//compute gradient (bottleneck is here)
		for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
			Int f_ind = it->first;
			Float f_val = it->second;
			for(vector<Int>::iterator it = act_uni_index.begin(); it != act_uni_index.end(); it++){
				Int k = *it;
				prod[k] += w[ f_ind ][k] * f_val ;
			}
		}
		
		//message=(E\beta-\alpha+\mu/\eta)
		if( msg_from_left != NULL)
			for(vector<Int>::iterator it = act_uni_index.begin(); it != act_uni_index.end(); it++){
				Int k = *it;
				prod[k] -= eta*msg_from_left[k];
			}
		
		if( msg_from_right != NULL )
			for(vector<Int>::iterator it = act_uni_index.begin(); it != act_uni_index.end(); it++){
				Int k = *it;
				prod[k] -= eta*msg_from_right[k];
			}
		
		//compute Dk
		for(Int ind = 0; ind < act_uni_index.size(); ind++){
			Int k = act_uni_index[ind];
			if( k != yi )
				Dk[ind] = prod[k];
			else
				Dk[ind] = prod[k] + Qii*C;
		}

		//sort according to D_k
		sort( Dk, Dk+act_uni_index.size(), greater<Float>() );
		
		//compute b by traversing D_k in descending order
		Float bb = Dk[0] - Qii*C;
		Int r;
		for( r=1; r<act_uni_index.size() && bb<r*Dk[r]; r++)
			bb += Dk[r];
		bb = bb / r;
		
		//record alpha new values
		for(vector<Int>::iterator it = act_uni_index.begin(); it != act_uni_index.end(); it++){
			Int k = *it;
			alpha_new[k] = min( (Float)((k!=yi)?0.0:C), (bb-prod[k])/Qii );
		}
		
		/*if( msg_from_left != NULL )
			delete[] msg_from_left;
		if( msg_from_right != NULL )
			delete[] msg_from_right;
		*/

		delete[] Dk;
	}

	void uni_search(Int i, Int n, Int t, vector<Int>& act_k_index){
		Seq* seq = data->at(n);
		Int yi = seq->labels[t];
		memset(prod, 0.0, sizeof(Float)*K);
		SparseVec* xi = seq->features[t];
		for (int j = 0; j < act_k_index.size(); j++){
			prod[act_k_index[j]] = -INFI;
		}
		prod[yi] = -INFI;
		Float th = -1.0;
		max_indices[0] = 0;
		for (SparseVec::iterator it = xi->begin(); it != xi->end(); it++){
			Float xij = it->second;
			Int j = it->first;
			Float* wj = w[j];
			for (int k = 0; k < K; k++){
				prod[k] += wj[k] * xij;
				if (prod[k] > prod[max_indices[0]]){
					max_indices[0] = k;
				}
			}
		}
		if (prod[max_indices[0]] < 0.0){
			for (Int r = 0; r < K; r++){
				Int k = rand()%K;
				if (prod[k] == 0.0){
					max_indices[0] = k;
					break;
				}
			}
		}
		if (prod[max_indices[0]] > th){
			act_k_index.push_back(max_indices[0]);
		}
	}
	
	
	void bi_subSolve(Int i, Int n, Int t, vector<Int>& act_bi_index, Float* beta_new){
		
		Int Ksq = K*K;
		Float* grad = new Float[Ksq];
		Float* Dk = new Float[act_bi_index.size()];
		
		//data
		Seq* seq = data->at(n);
		Int yi = seq->labels[t];
		Int yj = seq->labels[t+1];
		Int yi_yj = yi*K + yj;

		//variable values
		Float* beta_i = beta[i];
		Float* beta_suml_i = beta_suml[i];
		Float* beta_sumr_i = beta_sumr[i];
		//Float* msg_from_left = messages[2*bi_index(n,t)+F_LEFT];
		//Float* msg_from_right = messages[2*bi_index(n,t)+F_RIGHT];
		Float* msg_from_left = new Float[K];
		Float* msg_from_right = new Float[K];
		memset(msg_from_left, 0.0, sizeof(Float)*K);
		memset(msg_from_right, 0.0, sizeof(Float)*K);
		Int i1 = uni_index(n,t);
		for(Int k=0;k<K;k++){
			msg_from_left[k]  = beta_suml_i[k]-alpha[ i1   ][k]   + mu[2*i+F_LEFT ][k];
			msg_from_right[k] = beta_sumr_i[k]-alpha[ i1+1 ][k] + mu[2*i+F_RIGHT][k];
		}
		
		
		//compute gradient
		Float Qii = (1.0+eta*K);
		for(vector<Int>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			//if( k1k2 != yi_yj )
			//	grad[k1k2] = 1.0 - Qii*beta_i[k1k2];
			//else
			Int k1k2 = *it;
			grad[k1k2] = -Qii*beta_i[k1k2];
		}
		
		for(vector<Int>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = *it;
			Int k2 = k1k2 % K;
			Int k1 = (k1k2 - k2)/K;
			grad[k1k2] += v[k1][k2];
		}
		//grad: message from left
		for(vector<Int>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = *it;
			Int k2 = k1k2 % K;
			Int k1 = (k1k2 - k2)/K;
			Float tmp = eta*msg_from_left[k1];
			grad[k1k2] += tmp;
		}
		//grad: message from right
		for(vector<Int>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = *it;
			Int k2 = k1k2 % K;
			grad[k1k2] += eta*msg_from_right[k2];
		}
		
		//compute Dk
		for(Int it = 0; it < act_bi_index.size(); it++){
			Int k1k2 = act_bi_index[it];
			if( k1k2 != yi_yj )
				Dk[it] = grad[k1k2];
			else
				Dk[it] = grad[k1k2] + Qii*C;
		}
		
		//sort according to D_k
		sort( Dk, Dk+act_bi_index.size(), greater<Float>() );
		
		//compute b by traversing D_k in descending order
		Float b = Dk[0] - Qii*C;
		Int r;
		for( r=1; r<act_bi_index.size() && b<r*Dk[r]; r++)
			b += Dk[r];
		b = b / r;
		
		//record alpha new values
		for(vector<Int>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			Int k1k2 = *it;
			beta_new[k1k2] = min( (Float)((k1k2!=yi_yj)?0.0:C), (b-grad[k1k2])/Qii );
		}

		delete[] msg_from_left;
		delete[] msg_from_right;
		delete[] grad;
		delete[] Dk;
	}
	
	void bi_search(Int i, Int n, Int t, vector<Int>& act_bi_index){
		
		memset(inside, 0, sizeof(bool)*K*K);
		for (vector<Int>::iterator it = act_bi_index.begin(); it != act_bi_index.end(); it++){
			inside[*it] = true;
		}
		
		Int il = uni_index(n, t);
		Int ir = uni_index(n, t+1);
		Int max_k1k2 = -1;
		Float max_val = -1e300;
		//compute messages and then sort in decreasing order
		Float* msg_to_left = new Float[K];
		Float* msg_to_right = new Float[K];
		memset(msg_to_left, 0.0, sizeof(Float)*K);
		memset(msg_to_right, 0.0, sizeof(Float)*K);
		Float* beta_suml_i = beta_suml[i];
		Float* beta_sumr_i = beta_sumr[i];
		Float* alpha_il = alpha[il];
		Float* alpha_ir = alpha[ir];
		Float* mu_il = mu[i*2];
		Float* mu_ir = mu[i*2+1];
		vector<Int> pos_left;
		vector<Int> pos_right;
		for (int k = 0; k < K; k++){
			msg_to_left[k] = beta_suml_i[k] - alpha_il[k] + mu_il[k];
			msg_to_right[k] = beta_sumr_i[k] - alpha_ir[k] + mu_ir[k];
			if (msg_to_left[k] > 0.0)
				pos_left.push_back(k);
			if (msg_to_right[k] > 0.0)
				pos_right.push_back(k);
		}
		//sort(left_index, left_index+K, ScoreComp(msg_to_left));
		//sort(right_index, right_index+K, ScoreComp(msg_to_right));
		
		//check area 1: msg_to_left > 0 and msg_to_right > 0
		for (vector<Int>::iterator it_l = pos_left.begin(); it_l != pos_left.end(); it_l++){
			int kl = *it_l;
			Float* v_kl = v[kl];
			Float msg_L_kl = msg_to_left[kl];
			for (vector<Int>::iterator it_r = pos_right.begin(); it_r != pos_right.end(); it_r++){
				int kr = *it_r;
				Float val = v_kl[kr] + msg_L_kl + msg_to_right[kr];
				if (val > max_val && !inside[kl*K+kr]){
					max_val = val;
					max_k1k2 = kl*K+kr;
				}
			}
		}
			
		//check area 2: msg_to_left > 0 and msg_to_right = 0
		for (vector<Int>::iterator it_l = pos_left.begin(); it_l != pos_left.end(); it_l++){
			Int k1 = *it_l;
			search_line(row_heap[k1], msg_to_left[k1], msg_to_right, max_val, max_k1k2, row_heap_size[k1], inside, col_dir);
		}
		
		//check area 3: msg_to_left = 0 and msg_to_right > 0
		for (vector<Int>::iterator it_r = pos_right.begin(); it_r != pos_right.end(); it_r++){
			Int k2 = *it_r;
			search_line(col_heap[k2], msg_to_right[k2], msg_to_left, max_val, max_k1k2, col_heap_size[k2], inside, row_dir);
		}
	
		//check area 4: msg_to_left <= 0 and msg_to_right <= 0
		search_matrix(v_heap, msg_to_left, msg_to_right, max_val, max_k1k2, v_heap_size, inside, K);
		
		if (max_val > 0.0){
			act_bi_index.push_back(max_k1k2);
		}
		
		for(Int k=0;k<K;k++){
			
			double msg_L = beta_suml[i][k] - alpha[il][k] + mu[i*2][k];
			if( msg_L > 1e-2 )
				pos_count+=1.0;
			else if( msg_L < -1e-2 )
				neg_count+=1.0;
			else
				zero_count+=1.0;

			double msg_R = beta_sumr[i][k] - alpha[ir][k] + mu[i*2+1][k];
			if( msg_R > 1e-2 )
				pos_count+=1.0;
			else if( msg_R < -1e-2 )
				neg_count+=1.0;
			else
				zero_count+=1.0;
		}
	}
	
	void bi_brute_force_search(Int i, Int n, Int t, vector<Int>& act_bi_index){
		Int il = uni_index(n, t);
		Int ir = uni_index(n, t+1);
		Int max_k1k2 = -1;
		Float max_val = -1e300;
		
		for(int k1 = 0; k1 < K; k1++){
			for (int k2 = 0; k2 < K; k2++){
				if (find(act_bi_index.begin(), act_bi_index.end(), k1*K+k2) != act_bi_index.end()){
					continue;
				}
				Float val = v[k1][k2];
				val += beta_suml[i][k1];
				val -= alpha[il][k1];
				
				val += beta_sumr[i][k2];
				val -= alpha[ir][k2];
				val += mu[i*2][k1];
				val += mu[i*2+1][k2];
				if (val > max_val){
					max_k1k2 = k1*K+k2;
					max_val = val;
				}
			}
		}
		if (max_val > 0.0){
			act_bi_index.push_back(max_k1k2);
		}
		
		for(Int k=0;k<K;k++){
			
			double msg_L = beta_suml[i][k] - alpha[il][k] + mu[i*2][k];
			if( msg_L > 1e-2 )
				pos_count+=1.0;
			else if( msg_L < -1e-2 )
				neg_count+=1.0;
			else
				zero_count+=1.0;

			double msg_R = beta_sumr[i][k] - alpha[ir][k] + mu[i*2+1][k];
			if( msg_R > 1e-2 )
				pos_count+=1.0;
			else if( msg_R < -1e-2 )
				neg_count+=1.0;
			else
				zero_count+=1.0;
		}
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
				for(Int k=0;k<K;k++)
					prod[k] = 0.0;
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
	
	Float train_acc_Viterbi(){
		
		Int hit=0;
		for(Int n=0;n<nSeq;n++){
			
			Seq* seq = data->at(n);
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
					for(Int k2=0;k2<K;k2++){
						 cand_val = tmp + v[k1][k2];
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
	
	
	void marginalize( Float* table, Direction j, Float* marg ){
		
		for(Int k=0;k<K;k++)
			marg[k] = 0.0;
		
		if( j == F_LEFT ){
			for(Int k1=0;k1<K;k1++){
				Int Kk1 = K*k1;
				for(Int k2=0;k2<K;k2++)
					marg[k1] += table[Kk1+k2];
			}
		}else if( j == F_RIGHT ){
			for(Int k1=0;k1<K;k1++){
				Int Kk1 = K*k1;
				for(Int k2=0;k2<K;k2++)
					marg[k2] += table[Kk1+k2];
			}
		}else{
			cerr << "unknown direction: " << j << endl;
			exit(0);
		}
	}

	Float dual_obj(){
		
		Float uni_obj = 0.0;
		for(Int j=0;j<D;j++){
			for(Int k=0;k<K;k++){
				uni_obj += w[j][k] * w[j][k];
			}
		}
		uni_obj/=2.0;
		
		for(Int n=0;n<nSeq;n++){
			Seq* seq = data->at(n);
			for(Int t=0; t<seq->T; t++){
				Int i = uni_index(n,t);
				Int yi = seq->labels[t];
				for(Int k=0;k<K;k++)
					if( k != yi ){
						uni_obj += alpha[i][k];
					}
			}
		}

		Float bi_obj = 0.0;
		for(Int j=0;j<K;j++){
			for(Int k=0;k<K;k++)
				bi_obj += v[j][k] * v[j][k];
		}
		bi_obj/=2.0;
			
		Float p_inf_ijk;
		Float* marg_ij = new Float[K];
		Float p_inf = 0.0;
		for(Int n=0;n<nSeq;n++){
			Seq* seq = data->at(n);
			for(Int t=0;t<seq->T-1;t++){
				Int i2 = bi_index(n,t);
				Int i1 = uni_index(n,t);
				for(Int j=0;j<NUM_DIRECT;j++){
					//Float* msg_ij = messages[2*i2+j];
					//Float* mu_ij = mu[2*i2+j];
					marginalize(beta[i2], (Direction)j, marg_ij);
					for(Int k=0;k<K;k++){
						p_inf_ijk = marg_ij[k] - alpha[i1+j][k];
						//p_inf_ijk = msg_ij[k] - mu_ij[k];
						p_inf += p_inf_ijk * p_inf_ijk;
					}
				}
			}
		}
		p_inf *= eta/2.0;
		delete[] marg_ij;

		return uni_obj + bi_obj + p_inf;
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
	Float** beta_suml;	
	Float** beta_sumr;	
	Float** w; //D*K primal variables for unigram factor
	Float** v; //K^2 primal variables for bigram factor
		
	Float** mu; // 2M*K Lagrangian Multipliers on consistency constraInts
	//Float** messages;// 2M*K message=(E*beta-alpha+\frac{1}{\eta}\mu)
	
	Float* h_left;
	Float* h_right;

	Int max_iter;
	Float eta;
	Float admm_step_size;

	//uni_search
	Int* max_indices;
	Float* prod;

	//bi_search
	ArrayHeap* row_heap;
	ArrayHeap* col_heap;
	ArrayHeap v_heap;
	Int* row_heap_size;
	Int* col_heap_size;
	Int v_heap_size;
	Int* col_index;
	Int* row_index;
	Int* v_index;
	Int* col_dir;
	Int* row_dir;
	bool* inside;
	bool using_brute_force;
};
