#include "util.h"
#include "multilabel.h"
#include <cassert>

typedef vector<pair<Int, Float>> PairVec;

class BCFWsolve{

	public:
		enum Direction {F_LEFT=0, F_RIGHT=1, NUM_DIRECT};

		BCFWsolve(Param* param){

			//Parse info from ChainProblem
			prob = param->prob;
			data = &(prob->data);
			N = data->size();
			D = prob->D;
			K = prob->K;

			C = param->C;
			eta = param->eta;
			max_iter = param->max_iter;
			admm_step_size = 0.0;

			//allocate dual variables
			act_alpha = new PairVec[N];
			for(Int i=0;i<N;i++){
				act_alpha[i].clear();
			}

			act_beta = new vector<pair<Int, Float*>>[N];
			for(Int i=0;i<N;i++){
				act_beta[i].clear(); 
			}
			/*beta = new Float**[N];
			for(Int n=0;n<N;n++){
				beta[n] = new Float*[K*K];
				for(Int kk=0;kk<K*K;kk++)
					beta[n][kk] = NULL;
				for(Int i=0;i<K;i++)
					for(Int j=i+1;j<K;j++){
						beta[n][i*K+j] = new Float[4];
						memset( beta[n][i*K+j], 0.0, sizeof(Float)*4 );
					}
			}*/
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
				for(Int k2=0;k2<K;k2++)
					v[k][k2] = 0.0;
			}

			//allocating Lagrangian Multipliers for consistency constraInts
			mu = new Float**[N]; //2 because of bigram
			//messages = new Float*[2*M];
			for(Int n=0;n<N;n++){
				mu[n] = new Float*[K*K];
				for(Int i=0;i<K;i++){
					for(Int j=i+1;j<K;j++){
						mu[n][i*K+j] = new Float[2];
						for(int d=0;d<NUM_DIRECT;d++){
							mu[n][i*K+j][d] = 0.0;
						}
					}
				}
			}

			//pre-allocate some algorithmic constants
			Q_diag = new Float[N];
			for(Int n=0;n<N;n++){
				Instance* ins = data->at(n);
				Q_diag[n] = (K-1)*eta; //there are K-1 consistency constraints associated with alpha_i
				for(SparseVec::iterator it=ins->feature.begin(); it!=ins->feature.end(); it++){
					Q_diag[n] += it->second * it->second;
				}
			}

			//model that traces w, v
			model = new Model(w, v, prob);
			
			// pre-process positive labels
			is_pos_label = new bool*[N]; 
			for (Int i = 0; i < N; i++){
				is_pos_label[i] = new bool[K];
				Instance* ins = data->at(i);
				sort(ins->labels.begin(), ins->labels.end(), less<Int>());
				if (ins->labels.size() > 1)
					assert(ins->labels[0] < ins->labels[1]);
				memset(is_pos_label[i], false, sizeof(bool)*K);
				for (Labels::iterator it_label = ins->labels.begin(); it_label != ins->labels.end(); it_label++){
					Int k = *it_label;
					is_pos_label[i][k] = true;
				}
			}

			//global cache
			grad = new Float[K*K*4];
			memset(grad, 0.0, sizeof(Float)*K*K*4);
			inside = new bool[K*K];
			memset(inside, false, sizeof(bool)*K*K);
		}

		~BCFWsolve(){

			//dual variables
			delete[] act_alpha;
			for (Int i = 0; i < N; i++){
				for (vector<pair<Int, Float*>>::iterator it = act_beta[i].begin(); it != act_beta[i].end(); it++){
					delete[] it->second;
				}
			}
			delete[] act_beta;

			//for(Int i=0;i<M;i++)
			//	delete[] beta[i];
			//delete[] beta;
			//primal variables
			for(Int j=0;j<D;j++)
				delete[] w[j];
			delete[] w;
			for(Int k=0;k<K;k++)
				delete[] v[k];
			delete[] v;
			//Lagrangian Multiplier for Consistency constarInts
			for(Int n=0;n<N;n++){
				for(Int i=0;i<K;i++){
					for(Int j=i+1;j<K;j++){

						delete[] mu[n][i*K+j];
					}
				}
				delete[] mu[n];
			}
			delete[] mu;
			//delete[] messages;

			//some constants
			delete Q_diag;

			//delete cache for positive labels
			for (Int i = 0; i < N; i++)
				delete[] is_pos_label[i];
			delete[] is_pos_label;

			//global cache
			delete[] grad;
			delete[] inside;
		}

		Model* solve(){

			Int* sample_index = new Int[N];
			for(Int i=0;i<N;i++)
				sample_index[i] = i;

			//BDMM main loop
			Float* alpha_new = new Float[K];
			Float** beta_new = new Float*[K*K];
			for(Int i=0;i<K;i++)
				for(Int j=i+1;j<K;j++)
					beta_new[i*K+j] = new Float[4];
	
			for (Int n = 0; n < N; n++){
				Instance* ins = data->at(n);
				for (Labels::iterator it_label = ins->labels.begin(); it_label != ins->labels.end(); it_label++){
					act_alpha[n].push_back(make_pair(*it_label, 0.0));
				}
			}	
		
			/*for (Int n = 0; n < N; n++){
				for (Int k = 0; k < K; k++){	
					act_alpha[n].push_back(make_pair(k, 0.0));
				}
			}*/
		
			for (Int n = 0; n < N; n++){
				Instance* ins = data->at(n);
				for (Labels::iterator it_label = ins->labels.begin(); it_label != ins->labels.end(); it_label++){
					for (Labels::iterator it_label2 = it_label+1; it_label2 != ins->labels.end(); it_label2++){
						Int i = *it_label, j = *it_label2;
						Float* temp_float = new Float[4];
						memset(temp_float, 0.0, sizeof(Float)*4);
						act_beta[n].push_back(make_pair(K*i+j, temp_float));
					}
				}
			}

			/*for (Int n = 0; n < N; n++){
				for (Int i = 0; i < K; i++)
					for (Int j = i+1; j < K; j++){
						Float* temp_float = new Float[4];
						memset(temp_float, 0.0, sizeof(Float)*4);	
						act_beta[n].push_back(make_pair(K*i+j, temp_float));
					}
			}*/
				
			Float p_inf;
			Float alpha_nnz = 0.0;
			Float beta_nnz = 0.0;
			for(Int iter=0;iter<max_iter;iter++){

				random_shuffle(sample_index, sample_index+N);
				//update unigram dual variables
				alpha_nnz = 0.0;
				for(Int r=0;r<N;r++){

					Int n = sample_index[r];	
					//cout << "uni search:enter" << endl;	
					uni_search(n, act_alpha[n]);
					//cout << "uni search:exit" << endl;	

					//subproblem solving
					uni_subSolve(n, alpha_new);

					//maIntain relationship between w and alpha
					Instance* ins = data->at(n);
					
					for(SparseVec::iterator it=ins->feature.begin(); it!=ins->feature.end(); it++){
						Int j = it->first;
						Float fval = it->second;
						Float* wj = w[j];
						for(PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
							Int k = it_alpha->first;
							wj[k] += fval * (alpha_new[k]-it_alpha->second);
						}
					}

					//update alpha
					for(PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
						it_alpha->second = alpha_new[it_alpha->first];
					}
					alpha_nnz += act_alpha[n].size();
				}
				alpha_nnz /= N;
			
				//cout << "alpha_done" << endl;	
				//update bigram dual variables	
				random_shuffle(sample_index, sample_index+N);
				beta_nnz = 0.0;
				for(Int r=0;r<N;r++){
					Int n = sample_index[r];

					//cout << "bi search:enter" << endl;
					bi_search(n, act_beta[n]);
					/*if (iter >= K*K){
						cout << "size=" << act_beta[n].size() << endl;
						assert(act_beta[n].size() >= K*(K-1)/2);
					}*/
					//cout << "bi search:exit" << endl;
				
					//subproblem solving
					bi_subSolve(n, beta_new);

					//maIntain relationship between v and beta
					/*Float** beta_n = beta[n];
					for(Int i=0;i<K;i++){
						Int Ki = K*i;
						for(Int j=i+1;j<K;j++)
							v[i][j] += beta_new[Ki+j][3] - beta_n[Ki+j][3];
					}*/
					vector<pair<Int, Float*>>& act_beta_n = act_beta[n];
					for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n.begin(); it_beta != act_beta_n.end(); it_beta++){
						Int offset = it_beta->first;
						Float* beta_nij = it_beta->second;
						Int i = offset / K, j = offset % K;
						v[i][j] += beta_new[offset][3] - beta_nij[3];
					}
					
					//update beta
					/*for(Int i=0;i<K;i++){
						Int Ki = K*i;
						for(Int j=i+1;j<K;j++){
							for(int y1y2=0;y1y2<4;y1y2++)
								beta_n[Ki+j][y1y2] = beta_new[Ki+j][y1y2];
						}
					}*/
					vector<pair<Int, Float*>> tmp_vec;
					for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n.begin(); it_beta != act_beta_n.end(); it_beta++){
						Int offset = it_beta->first;
						Float* beta_nij = it_beta->second;
						for (Int d = 0; d < 4; d++){
							beta_nij[d] = beta_new[offset][d];
						}
						if (fabs(beta_nij[3]) > 1e-12){
							tmp_vec.push_back(make_pair(offset, beta_nij));
						} else {
							delete[] beta_nij;
						}
					}
					act_beta_n = tmp_vec;
					beta_nnz += act_beta_n.size();
				}
				beta_nnz /= N;

				//ADMM update (enforcing consistency)
				/*
				Float* mu_ij;
				Float p_inf_nij;
				p_inf = 0.0;
				for(Int n=0;n<N;n++){
					Instance* ins = data->at(n);
					Float** beta_n = beta[n];
					Float* alpha_n = new Float[K];
					memset(alpha_n, 0.0, sizeof(Float)*K);
					PairVec* ever_act_alpha_n = &(act_alpha[n]);
				//	for (PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
				//		alpha_n[it_alpha->first] = it_alpha->second;	
				//	}
				//	for(Int i=0;i<K;i++){
				//		Int Ki = K*i;
				//		for(Int j=i+1;j<K;j++){
				//			Float* mu_ij = mu[n][Ki+j];
				//			//left
				//			p_inf_nij = beta_n[Ki+j][2]+beta_n[Ki+j][3]-alpha_n[i];
				//			mu_ij[F_LEFT] += admm_step_size*p_inf_nij;
				//			p_inf += fabs(p_inf_nij);
				//			//right
				//			p_inf_nij = beta_n[Ki+j][1]+beta_n[Ki+j][3]-alpha_n[j];
				//			mu_ij[F_RIGHT] += admm_step_size*p_inf_nij;
				//			p_inf += fabs(p_inf_nij);
				//		}
				//	}

					// adds act_beta
					for(PairVec::iterator it_ever = ever_act_alpha_n.begin(); it_ever != ever_act_alpha_n.end(); it_ever++){
						Int i = it_ever->first;
						for(PairVec::iterator it_ever2 = it_ever+1; it_ever2 != ever_act_alpha_n.end(); it_ever2++){
							Int j = it_ever2->first;
							mu_ij[F_LEFT] -= alpha_n[i];
							mu_ij[F_RIGHT] -= alpha_n[i];
						}
					}
				}
				p_inf /= (N);
				*/
				/*double beta_nnz=0.0;
				for(Int n=0;n<N;n++){
					for(Int i=0;i<K;i++){
						Int Ki = K*i;
						for(Int j=i+1; j<K; j++){
							for(Int d=0;d<4;d++){
								if( fabs(beta[n][Ki+j][d]) > 1e-6 )
									beta_nnz+=1.0;
							}
						}
					}
				}
				beta_nnz /= N;

				double alpha_nnz=0.0;
				for(Int n=0;n<N;n++){
					alpha_nnz += act_alpha[n].size();
				}
				alpha_nnz/=N;
				*/
				cerr << "i=" << iter << ", a_nnz=" << alpha_nnz << ", b_nnz=" << beta_nnz;
					//<< ", infea=" << p_inf <<  ", d_obj=" << dual_obj();// << ", uAcc=" << train_acc_unigram();
				if(iter%10==9)
					cerr << ", Acc=" << train_acc_joint();
				cerr << endl;
			}
			

			delete[] sample_index;
			delete[] alpha_new;
			for(Int i=0;i<K;i++)
				for(Int j=i+1;j<K;j++)
					delete[] beta_new[i*K+j];
			delete[] beta_new;
			return new Model(w,v,prob);
		}

	private:

		void uni_subSolve(Int n, Float* alpha_new){ //solve i-th unigram factor

			//cout << "uni_subSolve: enter" << endl;
			Instance* ins = data->at(n);
			Float Qii = Q_diag[n];
			PairVec* act_alpha_n = &(act_alpha[n]);
			//cout << "uni_subSolve: enter1" << endl;
			vector<pair<Int, Float*>>* act_beta_n = &(act_beta[n]);
			Float* alpha_n = new Float[K];
			memset(alpha_n, 0.0, sizeof(Float)*K);
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				alpha_n[it_alpha->first] = it_alpha->second;
			}
			//cout << "uni_subSolve: enter2" << endl;
			Float** beta_n = new Float*[K*K];
			/*for (Int i = 0; i < K; i++)
				for (Int j = i+1; j < K; j++){
					beta_n[K*i+j] = new Float[4];
					memset(beta_n[K*i+j], 0.0, sizeof(Float)*4);
				}*/
			memset(inside, false, sizeof(bool)*K*K);
			//cout << "beta_n[offset] initialize" << endl;
			for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n->begin(); it_beta != act_beta_n->end(); it_beta++){
				Int offset = it_beta->first;
				beta_n[offset] = it_beta->second;
				inside[offset] = true;
			}
			//cout << "beta_n[offset] exit" << endl;
			Float** mu_n = mu[n];
			
			bool* is_pos_label_n = is_pos_label[n];
			
			Float* grad = new Float[K];
			memset(grad, 0.0, sizeof(Float)*K);
			for(SparseVec::iterator it=ins->feature.begin(); it!=ins->feature.end(); it++){
				Float* wj = w[it->first];
				for(Int k=0;k<K;k++){
					grad[k] += wj[k]*it->second;
				}
			}
			for(Int k=0;k<K;k++){
				if( !is_pos_label_n[k] )
					grad[k] += 1.0;
				else
					grad[k] += -1.0;
			}
			
			//messages
			/*for(Int i=0;i<K;i++){
				Int Ki = K*i;
				for(Int j=i+1;j<K;j++){//from each bigram
					Float* beta_nij = beta_n[Ki+j];
			*/
			Float msg_L, msg_R;
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				Int i = it_alpha->first;
				Float alpha_ni = it_alpha->second;
				Int Ki = K*i;
				for (PairVec::iterator it_alpha2 = it_alpha+1; it_alpha2 != act_alpha_n->end(); it_alpha2++){
					Int j = it_alpha2->first;
					Float alpha_nj = it_alpha2->second;
					Int offset = Ki + j;
					if (inside[offset]){
						//cout << "uni subsolve: before mu, offset=" << offset << endl;
						msg_L = beta_n[offset][2] + beta_n[offset][3] - alpha_ni + mu_n[offset][F_LEFT];
						msg_R = beta_n[offset][1] + beta_n[offset][3] - alpha_nj + mu_n[offset][F_RIGHT];
					} else {
						//cout << "bi search: before mu, offset=" << offset << endl;
						Float tmp_L = alpha_ni - mu_n[offset][F_LEFT];
						Float tmp_R = alpha_nj - mu_n[offset][F_RIGHT];
						//cout << "bi search: after mu" << endl;
						if (is_pos_label_n[i])
							tmp_L -= C;
						if (is_pos_label_n[j])
							tmp_R -= C;
						//solve problem: min (beta^{10} - tmp_L)^2 + (beta^{01} - tmp_R)^2; s.t. beta^{01}, beta^{10} \in [-C, 0], beta^{01} + beta^{10} \in [-C, 0]			
						msg_L = - tmp_L;
						msg_R = - tmp_R;
						if (tmp_L <= 0.0 && tmp_R <= 0.0){
							if (tmp_L + tmp_R > -C){
								Float t = 0.5*(tmp_L + tmp_R + C);
								tmp_L -= t;
								tmp_R -= t;
							}
						}
						msg_L += min(max(tmp_L, -(Float)C), 0.0);
						msg_R += min(max(tmp_R, -(Float)C), 0.0);
					}
					grad[i] -= eta * msg_L;
					
					grad[j] -= eta * msg_R;
					
					
					//bigram to left
					
					/*Float beta_nij10 = alpha_n[i] - beta_nij[3] - mu_n[Ki+j][F_LEFT];
					if (is_pos_label[i] && !is_pos_label[j]){
						//beta_nij10 \in [0, C]
						beta_nij10 = min(max(beta_nij10, 0.0), (Float)C);
					} else {
						//beta_nij10 \in [-C, 0]
						beta_nij10 = min(max(beta_nij10, -(Float)C), 0.0);
					}
					grad[i] -= eta*( beta_nij10 + beta_nij[3] - alpha_n[i] + mu_n[Ki+j][F_LEFT] );
					
					//bigram to right
					
					Float beta_nij01 = alpha_n[j] - beta_nij[3] - mu_n[Ki+j][F_RIGHT];
					if (!is_pos_label[i] && is_pos_label[j]){
						//beta_nij01 \in [0, C]
						beta_nij01 = min(max(beta_nij01, 0.0), (Float)C);
					} else {
						//beta_nij01 \in [-C, 0]
						beta_nij01 = min(max(beta_nij01, -(Float)C), 0.0);
					}
					grad[j] -= eta*( beta_nij01 + beta_nij[3] - alpha_n[j] + mu_n[Ki+j][F_RIGHT] );
					*/
				}
			}


			Float* U = new Float[K];
			Float* L = new Float[K];
			for(Int k=0;k<K;k++){
				U[k] = 0.0;
				L[k] = -C;
			}
			
			for(vector<int>::iterator it=ins->labels.begin(); it!=ins->labels.end(); it++){
				U[*it] = C;
				L[*it] = 0.0;
			}

			for(PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				Int k = it_alpha->first;
				alpha_new[k] = min( max( it_alpha->second - grad[k]/Qii, L[k] ), U[k] );
			}
		
			delete[] grad;	
			delete[] U;
			delete[] L;
		}

		void uni_search(Int n, vector<pair<Int, Float>>& act_alpha_n){
			Float* prod = new Float[K];
			memset(prod, 0.0, sizeof(Float)*K);
			Instance* ins = data->at(n);
			for (Labels::iterator it = ins->labels.begin(); it != ins->labels.end(); it++){
				prod[*it] = -INFI;
			}
			
			for (PairVec::iterator it = act_alpha_n.begin(); it != act_alpha_n.end(); it++){
				prod[it->first] = -INFI;
			}
			
			for (SparseVec::iterator it = ins->feature.begin(); it != ins->feature.end(); it++){
				Int j = it->first;
				Float xij = it->second;
				Float* wj = w[j];
				for (Int k = 0; k < K; k++){
					prod[k] += wj[k]*xij;
				}
			}
			int max_index = 0;
			for (Int k = 0; k < K; k++){
				if (prod[k] > prod[max_index])
					max_index = k;
			}
			if (prod[max_index] > -1){
				act_alpha_n.push_back(make_pair(max_index, 0.0));
				sort(act_alpha_n.begin(), act_alpha_n.end(), less<pair<Int, Float>>()) ;
			}
			delete[] prod;
			assert(act_alpha_n.size() <= K);
		}	

		void bi_subSolve(Int n, Float** beta_new){

			Int Ksq = K*K;
			Float* Dk = new Float[Ksq*4];

			//data
			Instance* ins = data->at(n);
			vector<pair<Int, Float*>>* act_beta_n = &(act_beta[n]);
			Float* alpha_n = new Float[K];
			memset(alpha_n, 0.0, sizeof(Float)*K);
			for (PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
				alpha_n[it_alpha->first] = it_alpha->second;
			}
			Float** mu_n = mu[n];
			//indicator of ground truth labels
			bool* is_pos_label_n = is_pos_label[n];

			Float Qii = (1.0+eta*2);
			for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n->begin(); it_beta != act_beta_n->end(); it_beta++){
					Int offset = it_beta->first;
					Float* beta_nij = it_beta->second;
					Int i = offset / K, j = offset % K;
					Int Ki = offset - j;
			/*for(Int i=0;i<K;i++){
				Int Ki = K*i;

				for(Int j=i+1; j<K; j++){
			*/
					//Int offset = (Ki+j)*4;
					//compute gradient
					for(int d=0;d<4;d++)
						//grad[offset+d] = - Qii*beta_n[Ki+j][d];
						grad[offset+d] = - Qii*beta_nij[d];						

					grad[offset + 3] += v[i][j];

					//Float beta_nij_left1_sum = beta_n[Ki+j][2]+beta_n[Ki+j][3];
					Float beta_nij_left1_sum = beta_nij[2]+beta_nij[3];
					grad[offset + 2] += eta*( beta_nij_left1_sum - alpha_n[i] + mu_n[Ki+j][F_LEFT] ); //2:10
					grad[offset + 3] += eta*( beta_nij_left1_sum - alpha_n[i] + mu_n[Ki+j][F_LEFT] ); //3:11

					Float beta_nij_right1_sum = beta_nij[1]+beta_nij[3];
					grad[offset + 1] += eta*( beta_nij_right1_sum - alpha_n[j] + mu_n[Ki+j][F_RIGHT] ); //1:01
					grad[offset + 3] += eta*( beta_nij_right1_sum - alpha_n[j] + mu_n[Ki+j][F_RIGHT] ); //1:11
					
					//compute Dk
					int d_nij = 2*is_pos_label_n[i] + is_pos_label_n[j];
					for(Int d=0;d<4;d++){

						Int ind = offset + d;
						if( d != d_nij )
							Dk[ind] = grad[ind];
						else
							Dk[ind] = grad[ind] + Qii*C;
					}
					
					//sort according to D_k
					sort( &(Dk[offset]), &(Dk[offset])+4,  greater<Float>() );
					//compute b by traversing D_k in descending order
					Float b = Dk[offset+0] - Qii*C;
					Int r;
					for( r=1; r<4 && b<r*Dk[offset+r]; r++)
						b += Dk[offset+r];
					b = b / r;
					
					//record beta new values
					for (int d=0;d<4;d++)
						beta_new[Ki+j][d] = min( (Float)((d!=d_nij)?0.0:C), (b-grad[offset+d])/Qii );
					
			//	}
			}
			
			delete[] alpha_n;
			delete[] Dk;
		}

		void bi_search(Int n, vector<pair<Int, Float*>>& act_beta_n){
			memset(grad, 0.0, sizeof(Float)*K*K*3);

			Int max_k1k2 = -1;
			Float max_val = -1e300;			

			memset(inside, false, sizeof(bool)*K*K);
			
			for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n.begin(); it_beta != act_beta_n.end(); it_beta++){
				Int offset = it_beta->first;
				inside[offset] = true;
			}

			/*for (vector<pair<Int, Float*>>& it_beta = act_beta_n.begin(); it_beta != act_beta_n.end(); it_beta++){
				Int offset = it_beta->first;
				Int i = offset / K;
				Int j = offset % K;
				Float* beta_nij = it_beta->second;
				grad[offset+3] = v[i][j] + beta_nij[2] + beta_nij[3]*2 + beta_nij[1];
				grad[offset+2] = beta_nij[2] + beta_nij[3];
				grad[offset+1] = beta_nij[1] + beta_nij[3];
			}*/
			
			Instance* ins = data->at(n);
			Labels* pos_labels = &(ins->labels);
			PairVec* act_alpha_n = &(act_alpha[n]);
			Float** mu_n = mu[n];

			bool* active = new bool[K];
			memset(active, false, sizeof(bool)*K);
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				active[it_alpha->first] = true;
			}

			//cout << "bi search:middle" << endl;
			bool* is_pos_label_n = is_pos_label[n];

			/*Float* msg_L = new Float[K*K];
			Float* msg_R = new Float[K*K];
			memset(msg_L, 0.0, sizeof(Float)*K*K);
			memset(msg_R, 0.0, sizeof(Float)*K*K);
			*/
			//area 1; y^n_i = y^n_j = 1
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				Int i = it_alpha->first;
				Float alpha_ni = it_alpha->second;
				Int Ki = K*i;
				for (PairVec::iterator it_alpha2 = it_alpha+1; it_alpha2 != act_alpha_n->end(); it_alpha2++){
					Int j = it_alpha2->first;
					Float alpha_nj = it_alpha2->second;
					Int offset = Ki + j;
					if (inside[offset])
						continue;
					//cout << "bi search: before mu, offset=" << offset << endl;
					Float tmp_L = alpha_ni - mu_n[offset][F_LEFT];
					Float tmp_R = alpha_nj - mu_n[offset][F_RIGHT];
					//cout << "bi search: after mu" << endl;
					if (is_pos_label_n[i])
						tmp_L -= C;
					if (is_pos_label_n[j])
						tmp_R -= C;
					//solve problem: min (beta^{10} - tmp_L)^2 + (beta^{01} - tmp_R)^2; s.t. beta^{01}, beta^{10} \in [-C, 0], beta^{01} + beta^{10} \in [-C, 0]			
					Float msg_L = - tmp_L;
					Float msg_R = - tmp_R;
					if (tmp_L <= 0.0 && tmp_R <= 0.0){
						if (tmp_L + tmp_R > -C){
							Float t = 0.5*(tmp_L + tmp_R + C);
							tmp_L -= t;
							tmp_R -= t;
						}
						//other wise tmp_L, tmp_R is already in simplex, no need for projection
					}
					msg_L += min(max(tmp_L, -(Float)C), 0.0);
					msg_R += min(max(tmp_R, -(Float)C), 0.0);
						
					Float val = v[i][j] + msg_L + msg_R;
					if (val > max_val){
						max_k1k2 = offset;
						max_val = val;
					}
					if (msg_L > max_val){
						max_k1k2 = offset;
						max_val = msg_L;
					}
					if (msg_R > max_val){
						max_k1k2 = offset;
						max_val = msg_R;
					}
				}
			}
			
			for (Int i = 0; i < K; i++){
				for (Int j = i+1; j < K; j++){
					if (active[i] && active[j])
						continue;
					if (inside[K*i+j])
						continue;
					if (v[i][j] > max_val){
						max_val = v[i][j];
						max_k1k2 = K*i+j;
					}
				}
			}

			/*for (Int i = 0; i < K; i++){
				for (Int j = i+1; j < K; j++){
					if (is_pos_label_n[i] && is_pos_label_n[j]) 
						continue;
					Float* beta_nij = beta_n;
					Float val = v[i][j] + eta;
				}
			}*/
			if (max_val > -1e300){
				Float* temp_float = new Float[4];
				memset(temp_float, 0.0, sizeof(Float)*4);
				act_beta_n.push_back(make_pair(max_k1k2, temp_float));
			}
			assert(act_beta_n.size() <= K*(K-1)/2);
		}

		Float train_acc_unigram(){

			Int* index = new Int[K];
			for(Int k=0;k<K;k++)
				index[k] = k;
			Float* prod = new Float[K];
			Int hit=0;
			for(Int n=0;n<N;n++){
				Instance* ins = data->at(n);

				SparseVec* xi = &(ins->feature);
				for(Int k=0;k<K;k++)
					prod[k] = 0.0;
				for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
					Float* wj = w[it->first];
					for(Int k=0;k<K;k++)
						prod[k] += wj[k]*it->second;
				}

				//compute top-1 precision
				/*sort( index, index+K, ScoreComp(prod) );
				  if( find(ins->labels.begin(), ins->labels.end(), index[0]) != ins->labels.end() )
				  hit++;
				  */
				//compute hamming loss
				int y;
				for(Int k=0;k<K;k++){
					if( find(ins->labels.begin(), ins->labels.end(), k) != ins->labels.end() )
						y = 1;
					else
						y = -1;

					if( y*prod[k] > 0.0 )
						hit++;
				}
			}
			//Float acc = (Float)hit/(N); //top-1 precision
			Float acc = (Float)hit/(N*K); //hamming loss

			delete[] index;
			delete[] prod;
			return acc;
		}
		
		Float train_acc_joint(){
			
			Int Ns = N/10;

			Int hit=0;
			Int* pred = new Int[K];
			for(Int n=0;n<Ns;n++){
				Instance* ins = data->at(n);
				model->LPpredict(ins, pred);
				for(Int k=0;k<K;k++){
					int yk = (find(ins->labels.begin(), ins->labels.end(), k) != ins->labels.end())? 1:0;
					if( pred[k] == yk )
						hit++;
				}
				
				if( n%10 ==0 )
					cerr << "." ;
			}

			delete[] pred;
			
			return (Float)hit/Ns/K;
		}

		Float dual_obj(){

			/*Float u_obj = 0.0;
			for(Int j=0;j<D;j++)
				for(Int k=0;k<K;k++)
					u_obj += w[j][k]*w[j][k];
			u_obj /= 2.0;

			for(Int n=0;n<N;n++){
				Instance* ins = data->at(n);
				for (PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
					Int k = it_alpha->first;
					if( find(ins->labels.begin(), ins->labels.end(), k) == ins->labels.end() )
						u_obj += it_alpha->second;
					else
						u_obj += -it_alpha->second;
				}
				//for(Int k=0;k<K;k++){
				//	if( find(ins->labels.begin(), ins->labels.end(), k) == ins->labels.end() )
				//		u_obj += alpha[n][k];
				//	else
				//		u_obj += - alpha[n][k];
				//}
			}

			Float bi_obj=0.0;
			for(Int i=0;i<K;i++)
				for(Int j=i+1;j<K;j++)
					bi_obj += v[i][j]*v[i][j];
			bi_obj/=2.0;
			
			
			Float pinf_nij;
			Float p_inf = 0.0;
			for(Int n=0;n<N;n++){
				Instance* ins = data->at(n);
				Float** beta_n = beta[n];
				Float* alpha_n = new Float[K];
				for (PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
					alpha_n[it_alpha->first] = it_alpha->second;
				}
				for(Int i=0;i<K;i++){
					Int Ki = K*i;
					for(Int j=i+1;j<K;j++){
						
						pinf_nij = beta_n[Ki+j][2]+beta_n[Ki+j][3]-alpha_n[i];
						p_inf += pinf_nij*pinf_nij;
						
						pinf_nij = beta_n[Ki+j][1]+beta_n[Ki+j][3]-alpha_n[j];
						p_inf += pinf_nij*pinf_nij;
					}
				}
			}
			p_inf *= eta/2.0;
			
			return u_obj + bi_obj + p_inf;*/
			return 0.0;
		}

		MultilabelProblem* prob;

		vector<Instance*>* data;
		Float C;
		Int N;
		Int D;
		Int K;

		// positive labels
		vector<Int>* pos_labels; 
		bool** is_pos_label;	

		//global cache
		Float* grad; // K*K*4 array
		bool* inside;

		Float* Q_diag;
		PairVec* act_alpha; //N*K dual variables for unigram factor
		
		vector<pair<Int, Float*>>* act_beta;
		Float*** beta; //N*M*4 dual variables for bigram factor (allocate N*K^2 instead of N*M for convenience)

		Float** w; //D*K primal variables for unigram factor
		Float** v; //K^2 primal variables for bigram factor

		Float*** mu; // N*K^2 * 2 Lagrangian Multipliers on consistency constraInts

		Int max_iter;
		Float eta;
		Float admm_step_size;
		
		Model* model;
};
