#include "util.h"
#include "multilabel.h"
#include <cassert>
#include <iomanip>
#include <ios>

typedef vector<pair<Int, Float>> PairVec;

extern double overall_time;

class BCFWsolve{

	public:
		enum Direction {F_LEFT=0, F_RIGHT=1, NUM_DIRECT};

		BCFWsolve(Param* param){

			//Parse info from ChainProblem
			prob = param->prob;
			heldout_prob = param->heldout_prob;
			data = &(prob->data);
			N = data->size();
			D = prob->D;
			K = prob->K;

			Float d = 0.0;
			for (vector<Instance*>::iterator it_data = data->begin(); it_data != data->end(); it_data++){
				Instance* ins = *it_data;
				d += ins->feature.size();	
			}
			d /= N;
			cerr << "d=" << d << endl;
			C = param->C;
			eta = param->eta;
			max_iter = param->max_iter;
			admm_step_size = param->admm_step_size;
			early_terminate = param->early_terminate;
			split_up_rate = param->split_up_rate;
			speed_up_rate = param->speed_up_rate;
			if( speed_up_rate==-1 )
				speed_up_rate = round( max( min( 5.0*D*K/(N*d)/C/log((Float)K), d/5.0), 1.0) );
			cerr << "r=" << speed_up_rate << endl;
			heldout_period = param->heldout_period;
			do_subSolve = param->do_subSolve;
			if (heldout_period == -1){
				heldout_period = 10;
			}

			if (early_terminate == -1){
				early_terminate = 10;
			}

			//allocate dual variables
			act_alpha = new PairVec[N];
			for(Int i=0;i<N;i++){
				act_alpha[i].clear();
			}

			act_beta = new vector<pair<Int, Float*>>[N];
			for(Int i=0;i<N;i++){
				act_beta[i].clear(); 
			}

			ever_act_alpha = new vector<Int>[N];
			
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
			mu = new vector<pair<Int, Float*>>[N]; //2 because of bigram
			for(Int n=0;n<N;n++){
				mu[n].clear();
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
				memset(is_pos_label[i], false, sizeof(bool)*K);
				for (Labels::iterator it_label = ins->labels.begin(); it_label != ins->labels.end(); it_label++){
					Int k = *it_label;
					is_pos_label[i][k] = true;
				}
			}

			//global cache
			grad = new Float[K];
			memset(grad, 0.0, sizeof(Float)*K);
			inside = new bool[K*K];
			memset(inside, false, sizeof(bool)*K*K);
			is_ever_active = new bool*[N];
			for (Int i = 0; i < N; i++){
				is_ever_active[i] = new bool[K];
				memset(is_ever_active[i], false, sizeof(bool)*K);
			}
			alpha_n = new Float[K];
			memset(alpha_n, 0.0, sizeof(Float)*K);
			beta_n = new Float*[K*K];
			for (Int ij = 0; ij < K*K; ij++){
				beta_n[ij] = NULL;
			}
			mu_n = new Float*[K*K];
			for (Int i = 0; i < K; i++){
				for (Int j = i+1; j < K; j++){
					Int offset = K*i+j;
					mu_n[offset] = new Float[2];
					mu_n[offset][0] = 0.0;
					mu_n[offset][1] = 0.0;
				}
			}
			
			//uni_search
			l1_norm = new Float[N];
			for (Int n = 0; n < N; n++){
				l1_norm[n] = 0.0;
				Instance* ins = data->at(n);
				for (SparseVec::iterator it = ins->feature.begin(); it != ins->feature.end(); it++){
					l1_norm[n] += fabs(it->second);
				}
			}

			using_brute_force_search = param->using_brute_force_search;
			if (using_brute_force_search){
				split_up_rate = 1;
				speed_up_rate = 1;
			}

			//maintain heap for v[i][j]
			v_heap_size = 0;
			v_heap = new pair<Float, Int>[(K*(K-1))/2];
			v_index = new Int[K*K];
			for (Int i = 0; i < K; i++){
				for (Int j = i+1; j < K; j++){
					Int ij = K*i+j;
					v_heap[v_heap_size] = make_pair(0.0, ij);
					v_index[ij] = v_heap_size++;
				}
			}
		}

		~BCFWsolve(){

			//dual variables
			delete[] act_alpha;
			delete[] ever_act_alpha;
			for (Int i = 0; i < N; i++){
				for (vector<pair<Int, Float*>>::iterator it = act_beta[i].begin(); it != act_beta[i].end(); it++){
					delete[] it->second;
				}
			}
			delete[] act_beta;

			//primal variables
			for(Int j=0;j<D;j++)
				delete[] w[j];
			delete[] w;
			for(Int k=0;k<K;k++)
				delete[] v[k];
			delete[] v;
			//Lagrangian Multiplier for Consistency constarInts
			
			for (Int n = 0; n < N; n++){
				for (vector<pair<Int, Float*>>::iterator it_m = mu[n].begin(); it_m != mu[n].end(); it_m++){
					delete[] it_m->second;
				}
				mu[n].clear();
			}
			delete[] mu;
			//delete[] messages;

			//uni_search
			delete[] l1_norm;

			//some constants
			delete Q_diag;

			//delete cache for positive labels
			for (Int i = 0; i < N; i++)
				delete[] is_pos_label[i];
			delete[] is_pos_label;

			//global cache
			delete[] grad;
			delete[] inside;
			for (Int i = 0; i < N; i++)
				delete[] is_ever_active[i];
			
			delete[] is_ever_active;
			delete[] alpha_n;
			delete[] beta_n;
			for (Int i = 0; i < K; i++){
				for (Int j = i+1; j < K; j++){
					Int offset = K*i+j;
					delete[] mu_n[offset];
				}
			}
			delete[] mu_n;
			//maintain heap for v[i][j]
			delete[] v_heap;
			delete[] v_index;
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
					ever_act_alpha[n].push_back(*it_label);
					is_ever_active[n][*it_label] = true;
				}
			}

			for (Int n = 0; n < N; n++){
				Instance* ins = data->at(n);
				for (Labels::iterator it_label = ins->labels.begin(); it_label != ins->labels.end(); it_label++){
					for (Labels::iterator it_label2 = it_label+1; it_label2 != ins->labels.end(); it_label2++){
						Int i = *it_label, j = *it_label2;
						if (i > j){
							Int temp = i; i = j; j = temp;
						}
						Float* temp_float = new Float[4];
						memset(temp_float, 0.0, sizeof(Float)*4);
						act_beta[n].push_back(make_pair(K*i+j, temp_float));
					}
				}
			}
				
			Float p_inf;
			Float alpha_nnz = 0.0;
			Float beta_nnz = 0.0;
			Float ever_alpha_nnz = 0.0;
			Float max_heldout_test_acc = 0.0;
			Int terminate_counting = 0;
			for(Int iter=0;iter<max_iter;iter++){

				random_shuffle(sample_index, sample_index+N);
				double uni_search_time = 0.0, uni_subSolve_time = 0.0, uni_maintain_time = 0.0;
				double bi_search_time = 0.0,  bi_subSolve_time = 0.0,  bi_maintain_time = 0.0;
				double admm_maintain_time = 0.0;
				//update unigram dual variables
				alpha_nnz = 0.0;
				ever_alpha_nnz = 0.0;	
				for(Int r=0;r<N;r++){

					Int n = sample_index[r];

					//search for active variables	
					uni_search_time -= omp_get_wtime();
					uni_search(n, act_alpha[n]);
					uni_search_time += omp_get_wtime();

					//subproblem solving
					uni_subSolve_time -= omp_get_wtime();
					if (do_subSolve)
						uni_subSolve(n, alpha_new);
					else
						uni_update(n, alpha_new, iter);
					uni_subSolve_time += omp_get_wtime();

					//maIntain relationship between w and alpha
					uni_maintain_time -= omp_get_wtime();
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

					//update and shrink alpha
					PairVec tmp_vec;
					bool* is_ever_active_n = is_ever_active[n];
					for(PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
						Int k = it_alpha->first;
						it_alpha->second = alpha_new[k];
						Float alpha_nk = it_alpha->second;
						if (fabs(alpha_nk)>1e-12 || is_pos_label[n][k]){
							tmp_vec.push_back(make_pair(k, alpha_nk));
							if (!is_ever_active_n[k]){
								for (vector<pair<Int, Float*>>::iterator it_b = act_beta[n].begin(); it_b != act_beta[n].end(); it_b++)
									inside[it_b->first] = true;
								for (vector<Int>::iterator it_e = ever_act_alpha[n].begin(); it_e != ever_act_alpha[n].end(); it_e++){
									Int i = *it_e, j = k;
									if (i > j){
										Int temp = i; i = j; j = temp;
									}
									Int offset = K*i+j;
									if (inside[offset]) continue;
									Float* temp_float = new Float[4];
									memset(temp_float, 0.0, sizeof(Float)*4);
									act_beta[n].push_back(make_pair(offset, temp_float));
								}
								for (vector<pair<Int, Float*>>::iterator it_b = act_beta[n].begin(); it_b != act_beta[n].end(); it_b++)
									inside[it_b->first] = false;
								is_ever_active_n[k] = true;
								ever_act_alpha[n].push_back(k);
							}
						}
					}
					act_alpha[n] = tmp_vec;

					alpha_nnz += act_alpha[n].size();
					ever_alpha_nnz += ever_act_alpha[n].size();
					uni_maintain_time += omp_get_wtime();
				}
				alpha_nnz /= N;
				ever_alpha_nnz /= N;
			
				//update bigram dual variables	
				random_shuffle(sample_index, sample_index+N);
				beta_nnz = 0.0;
				for(Int r=0;r<N;r++){
					Int n = sample_index[r];
					
					//search for active variables
					bi_search_time -= omp_get_wtime();
					if (using_brute_force_search){
						bi_brute_force_search(n, act_beta[n]);
					} else {
						bi_search(n, act_beta[n]);
					}
					bi_search_time += omp_get_wtime();
				
					//subproblem solving
					bi_subSolve_time -= omp_get_wtime();
					if (do_subSolve){
						bi_subSolve(n, beta_new);
					} else {
						bi_update(n, beta_new, iter);
					}
					bi_subSolve_time += omp_get_wtime();

					//maIntain relationship between v and beta
					bi_maintain_time -= omp_get_wtime();
					vector<pair<Int, Float*>>& act_beta_n = act_beta[n];
					for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n.begin(); it_beta != act_beta_n.end(); it_beta++){
						Int offset = it_beta->first;
						Float* beta_nij = it_beta->second;
						Float delta_beta = beta_new[offset][3] - beta_nij[3];
						if (fabs(delta_beta) < 1e-12)
							continue;
						Int i = offset / K, j = offset % K;
						v[i][j] += delta_beta;
						Int v_ind = v_index[offset];
						assert(v_heap[v_ind].second == offset);
						v_heap[v_ind].first += delta_beta;
						if (delta_beta > 0.0){
							siftUp(v_heap, v_ind, v_index);	
						} else {
							siftDown(v_heap, v_ind, v_index, v_heap_size);	
						}
					}

					//update and shrink beta
					vector<pair<Int, Float*>> tmp_vec;
					for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n.begin(); it_beta != act_beta_n.end(); it_beta++){
						Int offset = it_beta->first;
						Int i = offset / K, j = offset % K;
						Float* beta_nij = it_beta->second;
						for (Int d = 0; d < 4; d++){
							beta_nij[d] = beta_new[offset][d];
						}
						if (fabs(beta_nij[3]) > 1e-12 || (is_ever_active[n][i] && is_ever_active[n][j]) ){
							tmp_vec.push_back(make_pair(offset, beta_nij));
						} else {
							delete[] beta_nij;
						}
					}	
					act_beta_n = tmp_vec;
					beta_nnz += act_beta_n.size();
					bi_maintain_time += omp_get_wtime();
				}
				beta_nnz /= N;
				
				//ADMM update (enforcing consistency)
				admm_maintain_time -= omp_get_wtime();
				p_inf = 0.0;
				bool* inside_mu = new bool[K*K];
				memset(inside_mu, 0.0, sizeof(bool)*K*K);
				for(Int n=0;n<N;n++){
					Instance* ins = data->at(n);
					bool* is_pos_label_n = is_pos_label[n];
					vector<Int>* ever_act_alpha_n = &(ever_act_alpha[n]);
					cache_mu_n(n);
					for (vector<pair<Int, Float*>>::iterator it_m = mu[n].begin(); it_m != mu[n].end(); it_m++){
						inside_mu[it_m->first] = true;
					}
					//memset(alpha_n, 0.0, sizeof(Float)*K);
					for (PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
						alpha_n[it_alpha->first] = it_alpha->second;
					}
					
					for (vector<pair<Int, Float*>>::iterator it_b = act_beta[n].begin(); it_b != act_beta[n].end(); it_b++){
						Int offset = it_b->first;
						inside[offset] = true;
						beta_n[offset] = it_b->second;
					}	
					
					for(vector<Int>::iterator it_ever = ever_act_alpha_n->begin(); it_ever != ever_act_alpha_n->end(); it_ever++){
						Int ii = *it_ever;
						for(vector<Int>::iterator it_ever2 = it_ever+1; it_ever2 != ever_act_alpha_n->end(); it_ever2++){
							Int i = ii;
							Int j = *it_ever2;
							if (i > j){
								Int temp = i; i = j; j = temp;
							}
							Int offset = K*i+j;
							Float* mu_nij = mu_n[offset];
							if (inside[offset]){
								Float delta_mu_l = beta_n[offset][2] + beta_n[offset][3] - alpha_n[i];
								Float delta_mu_r = beta_n[offset][1] + beta_n[offset][3] - alpha_n[j];
								p_inf += delta_mu_l * delta_mu_l;
								p_inf += delta_mu_r * delta_mu_r;
								mu_nij[F_LEFT] += admm_step_size*(delta_mu_l);
								mu_nij[F_RIGHT] += admm_step_size*(delta_mu_r);
							} else {	
								Float beta_nij01, beta_nij10;
								recover_beta(n, i, j, alpha_n, mu_n, beta_nij10, beta_nij01);
								Float delta_mu_l = beta_nij10 - alpha_n[i];
								Float delta_mu_r = beta_nij01 - alpha_n[j];
								p_inf += delta_mu_l * delta_mu_l;
								p_inf += delta_mu_r * delta_mu_r;
								mu_nij[F_LEFT] += admm_step_size*(delta_mu_l);
								mu_nij[F_RIGHT] += admm_step_size*(delta_mu_r);
							}
							if (!inside_mu[offset] && (fabs(mu_nij[F_LEFT]) > 1e-12 || fabs(mu_nij[F_RIGHT]) > 1e-12) ){
								inside_mu[offset] = true;
								Float* temp_float = new Float[2];
								temp_float[F_LEFT] = mu_nij[F_LEFT];
								temp_float[F_RIGHT] = mu_nij[F_RIGHT];
								mu[n].push_back(make_pair(offset, temp_float));
							}
						}
					}
					bool* is_ever_active_n = is_ever_active[n];
					for (vector<pair<Int, Float*>>::iterator it_beta = act_beta[n].begin(); it_beta != act_beta[n].end(); it_beta++){
						Int offset = it_beta->first;
						Int i = offset / K, j = offset % K;
						Float* mu_nij = mu_n[offset];
						if (is_ever_active_n[i] && is_ever_active_n[j]){
							continue;
						}
						Float* beta_nij = it_beta->second;
						Float delta_mu_l = beta_nij[2] + beta_nij[3] - alpha_n[i];
						Float delta_mu_r = beta_nij[1] + beta_nij[3] - alpha_n[j];
						p_inf += delta_mu_l * delta_mu_l;
						p_inf += delta_mu_r * delta_mu_r;
						mu_nij[F_LEFT] += (delta_mu_l) * admm_step_size;
						mu_nij[F_RIGHT] += (delta_mu_r) * admm_step_size;
						if (!inside_mu[offset] && ( fabs(mu_nij[F_LEFT]) > 1e-12 || fabs(mu_nij[F_RIGHT]) > 1e-12 ) ){
							assert(i < j);
							inside_mu[offset] = true;
							Float* temp_float = new Float[2];
							temp_float[F_LEFT] = mu_nij[F_LEFT];
							temp_float[F_RIGHT] = mu_nij[F_RIGHT];
							mu[n].push_back(make_pair(offset, temp_float));
						}
					}

					for (vector<pair<Int, Float*>>::iterator it_beta = act_beta[n].begin(); it_beta != act_beta[n].end(); it_beta++){
						Int offset = it_beta->first;
						beta_n[offset] = NULL;
						inside[offset] = false;
					}
					
					for (PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
						alpha_n[it_alpha->first] = 0.0;
					}
					
					for (vector<pair<Int, Float*>>::iterator it_m = mu[n].begin(); it_m != mu[n].end(); it_m++){
						Int offset = it_m->first;
						it_m->second[0] = mu_n[offset][0];
						it_m->second[1] = mu_n[offset][1];
						inside_mu[offset] = false;
					}

					clean_mu_n(n);
				}
				delete[] inside_mu;
				p_inf *= (eta/2);
				admm_maintain_time += omp_get_wtime();
				
				cerr << "i=" << iter << ", a_nnz=" << alpha_nnz << ", ever_a_nnz=" << ever_alpha_nnz << ", b_nnz=" << beta_nnz;
				cerr << ", uni_search=" << uni_search_time << ", uni_subSolve=" << uni_subSolve_time << ", uni_maintain_time=" << uni_maintain_time;
				cerr << ", bi_search="  << bi_search_time  << ", bi_subSolve="  << bi_subSolve_time  << ", bi_maintain_time="  << bi_maintain_time;
				cerr << ", admm_maintain="  << admm_maintain_time;
				cerr << ", area4=" << (Float)mat_top/mat_bottom;
				cerr << ", infea=" << p_inf <<  ", d_obj=" << dual_obj();
				mat_top = mat_bottom = 0;
				if((iter+1)%heldout_period==0){
					if (heldout_prob == NULL){
						overall_time += omp_get_wtime();
						cerr << ", train Acc=" << train_acc_joint();
						overall_time -= omp_get_wtime();
					} else {
						overall_time += omp_get_wtime();
						Float heldout_test_acc = heldout_acc_joint();	
						cerr << ", heldout Acc=" << heldout_test_acc;
						overall_time -= omp_get_wtime();
						if (heldout_test_acc > max_heldout_test_acc){
							max_heldout_test_acc = heldout_test_acc;
							terminate_counting = 0;
						} else {
							cerr << " (" << (++terminate_counting) << "/" << (early_terminate) << ")";
							if (terminate_counting == early_terminate){
								//TODO should write best acc model
								cerr << endl;
								break;	
							}
						}
					}
				}
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

		void uni_subSolve(Int n, Float* alpha_new){ //solve n-th unigram factor

			Instance* ins = data->at(n);
			Float Qii = Q_diag[n];
			PairVec* act_alpha_n = &(act_alpha[n]);
			vector<Int>* ever_act_alpha_n = &(ever_act_alpha[n]);
			vector<pair<Int, Float*>>* act_beta_n = &(act_beta[n]);
			//memset(alpha_n, 0.0, sizeof(Float)*K);
			//memset(grad, 0.0, sizeof(Float)*K);
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				alpha_n[it_alpha->first] = it_alpha->second;
			}
			
			for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n->begin(); it_beta != act_beta_n->end(); it_beta++){
				Int offset = it_beta->first;
				beta_n[offset] = it_beta->second;
				inside[offset] = true;
			}
			cache_mu_n(n);
			
			bool* is_pos_label_n = is_pos_label[n];
			bool* is_ever_active_n = is_ever_active[n];
			
			for(SparseVec::iterator it=ins->feature.begin(); it!=ins->feature.end(); it++){
				Float* wj = w[it->first];
				for(PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
					Int k = it_alpha->first;
					grad[k] += wj[k]*it->second;
				}
			}

			for(PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				Int k = it_alpha->first;
				if( !is_pos_label_n[k] )
					grad[k] += 1.0;
				else
					grad[k] += -1.0;
			}
			
			Float msg_L, msg_R;
			for (vector<Int>::iterator it_alpha = ever_act_alpha_n->begin(); it_alpha != ever_act_alpha_n->end(); it_alpha++){
				Int ii = *it_alpha;
				for (vector<Int>::iterator it_alpha2 = it_alpha+1; it_alpha2 != ever_act_alpha_n->end(); it_alpha2++){
					Int i = ii;
					Int j = *it_alpha2;
					if (i > j){
						Int temp = i;
						i = j;
						j = temp;
					}
					Float alpha_ni = alpha_n[i];
					Int Ki = K*i;
					Float alpha_nj = alpha_n[j];
					Int offset = Ki + j;
					assert(mu_n[offset] != NULL);
					if (inside[offset]){
						msg_L = beta_n[offset][2] + beta_n[offset][3] - alpha_n[i] + mu_n[offset][F_LEFT];
						msg_R = beta_n[offset][1] + beta_n[offset][3] - alpha_n[j] + mu_n[offset][F_RIGHT];
					} else {	
						Float beta_nij01, beta_nij10;
						recover_beta(n, i, j, alpha_n, mu_n, beta_nij10, beta_nij01);
						msg_L = beta_nij10 - alpha_n[i] + mu_n[offset][F_LEFT];
						msg_R = beta_nij01 - alpha_n[j] + mu_n[offset][F_RIGHT];
					}
					grad[i] -= eta * msg_L;
					grad[j] -= eta * msg_R;
				}
			}
		
			for (vector<pair<Int, Float*>>::iterator it_b = act_beta[n].begin(); it_b != act_beta[n].end(); it_b++){
				Int offset = it_b->first;
				Int i = offset / K, j = offset % K;
				if (!is_ever_active_n[i] || !is_ever_active_n[j]){
					msg_L = beta_n[offset][2] + beta_n[offset][3] - alpha_n[i] + mu_n[offset][F_LEFT];
					msg_R = beta_n[offset][1] + beta_n[offset][3] - alpha_n[j] + mu_n[offset][F_RIGHT];
					grad[i] -= eta * msg_L;
					grad[j] -= eta * msg_R;
				}
			}
		
			for(PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				Int k = it_alpha->first;
				if (is_pos_label_n[k])
					alpha_new[k] = min( max( it_alpha->second - grad[k]/Qii, 0.0 ), (Float)C );
				else
					alpha_new[k] = min( max( it_alpha->second - grad[k]/Qii, -(Float)C ), 0.0 );
			}
	
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++)
				alpha_n[it_alpha->first] = 0.0;

			for (vector<Int>::iterator it_e = ever_act_alpha_n->begin(); it_e != ever_act_alpha_n->end(); it_e++)
				grad[*it_e] = 0.0;
			
			for (vector<pair<Int, Float*>>::iterator it_b = act_beta[n].begin(); it_b != act_beta[n].end(); it_b++){
				Int offset = it_b->first;
				Int i = offset / K, j = offset % K;
				inside[offset] = false;
				beta_n[offset] = NULL;
				grad[i] = 0.0; grad[j] = 0.0;
			}
			clean_mu_n(n);
		}

		void uni_update(Int n, Float* alpha_new, Int iter){ //solve n-th unigram factor

			Instance* ins = data->at(n);
			Float Qii = Q_diag[n];
			PairVec* act_alpha_n = &(act_alpha[n]);
			vector<Int>* ever_act_alpha_n = &(ever_act_alpha[n]);
			vector<pair<Int, Float*>>* act_beta_n = &(act_beta[n]);
			//memset(alpha_n, 0.0, sizeof(Float)*K);
			//memset(grad, 0.0, sizeof(Float)*K);
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				alpha_n[it_alpha->first] = it_alpha->second;
			}
			
			for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n->begin(); it_beta != act_beta_n->end(); it_beta++){
				Int offset = it_beta->first;
				beta_n[offset] = it_beta->second;
				inside[offset] = true;
			}
			cache_mu_n(n);
			
			bool* is_pos_label_n = is_pos_label[n];
			bool* is_ever_active_n = is_ever_active[n];
			
			for(SparseVec::iterator it=ins->feature.begin(); it!=ins->feature.end(); it++){
				Float* wj = w[it->first];
				for(PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
					Int k = it_alpha->first;
					grad[k] += wj[k]*it->second;
				}
			}

			for(PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				Int k = it_alpha->first;
				if( !is_pos_label_n[k] )
					grad[k] += 1.0;
				else
					grad[k] += -1.0;
			}
		
			Float msg_L, msg_R;
			for (vector<Int>::iterator it_alpha = ever_act_alpha_n->begin(); it_alpha != ever_act_alpha_n->end(); it_alpha++){
				Int ii = *it_alpha;
				for (vector<Int>::iterator it_alpha2 = it_alpha+1; it_alpha2 != ever_act_alpha_n->end(); it_alpha2++){
					Int i = ii;
					Int j = *it_alpha2;
					if (i > j){
						Int temp = i;
						i = j;
						j = temp;
					}
					Float alpha_ni = alpha_n[i];
					Int Ki = K*i;
					Float alpha_nj = alpha_n[j];
					Int offset = Ki + j;
					assert(mu_n[offset] != NULL);
					if (inside[offset]){
						msg_L = beta_n[offset][2] + beta_n[offset][3] - alpha_n[i] + mu_n[offset][F_LEFT];
						msg_R = beta_n[offset][1] + beta_n[offset][3] - alpha_n[j] + mu_n[offset][F_RIGHT];
					} else {
						Float beta_nij01, beta_nij10;
						recover_beta(n, i, j, alpha_n, mu_n, beta_nij10, beta_nij01);
						msg_L = beta_nij10 - alpha_n[i] + mu_n[offset][F_LEFT];
						msg_R = beta_nij01 - alpha_n[j] + mu_n[offset][F_RIGHT];
					}
					grad[i] -= eta * msg_L;
					grad[j] -= eta * msg_R;
				}
			}
			
			for (vector<pair<Int, Float*>>::iterator it_b = act_beta[n].begin(); it_b != act_beta[n].end(); it_b++){
				Int offset = it_b->first;
				Int i = offset / K, j = offset % K;
				if (!is_ever_active_n[i] || !is_ever_active_n[j]){
					msg_L = beta_n[offset][2] + beta_n[offset][3] - alpha_n[i] + mu_n[offset][F_LEFT];
					msg_R = beta_n[offset][1] + beta_n[offset][3] - alpha_n[j] + mu_n[offset][F_RIGHT];
					grad[i] -= eta * msg_L;
					grad[j] -= eta * msg_R;
				}
			}

			//compute best extreme point of FW
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				Int k = it_alpha->first;
				if (is_pos_label_n[k]){
					if (grad[k] > 0.0){
						alpha_new[k] = 0.0;
					} else {
						alpha_new[k] = C;
					}
				} else {
					if (grad[k] > 0.0){
						alpha_new[k] = -C;
					} else {
						alpha_new[k] = 0.0;
					}
				}
			}	

			//compute gamma
			Float gamma = 0.0, up = 0.0, down = 0.0;
			for (PairVec::iterator it_a = act_alpha_n->begin(); it_a != act_alpha_n->end(); it_a++){
				Int k = it_a->first;
				Float delta_alpha = alpha_new[k] - it_a->second;
				up -= grad[k] * delta_alpha; 
				down += (Qii) * delta_alpha * delta_alpha;
			}

			if (fabs(down) > 1e-12)
				gamma = up / down;
			if (gamma > 1.0)
				gamma = 1.0;
			if (gamma < 0.0)
				gamma = 0.0;
			
			//compute alpha new
			for (PairVec::iterator it_a = act_alpha_n->begin(); it_a != act_alpha_n->end(); it_a++){
				Int k = it_a->first;
				alpha_new[k] = (1-gamma)*it_a->second + gamma*alpha_new[k];
			}

			// clean up
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++)
				alpha_n[it_alpha->first] = 0.0;

			for (vector<Int>::iterator it_e = ever_act_alpha_n->begin(); it_e != ever_act_alpha_n->end(); it_e++)
				grad[*it_e] = 0.0;
			
			for (vector<pair<Int, Float*>>::iterator it_b = act_beta[n].begin(); it_b != act_beta[n].end(); it_b++){
				Int offset = it_b->first;
				Int i = offset / K, j = offset % K;
				inside[offset] = false;
				beta_n[offset] = NULL;
				grad[i] = 0.0; grad[j] = 0.0;
			}
			clean_mu_n(n);
		}

		void uni_search(Int n, vector<pair<Int, Float>>& act_alpha_n){
			Int rand_interval = rand() % split_up_rate;
			Int interval_length = K / split_up_rate;
			Int range_l = interval_length * rand_interval;
			Int range_r = interval_length * (rand_interval + 1);
			Instance* ins = data->at(n);
			if (rand_interval + 1 == split_up_rate)
				range_r = K;
			Int m = ins->feature.size() / speed_up_rate;
			if (ins->feature.size() < speed_up_rate){
				m = ins->feature.size();
			}
			vector<Float> rand_nums;
			for (Int tt = 0; tt < m; tt++){
				Float random_float = rand();
				random_float /= (Float)RAND_MAX;
				random_float *= l1_norm[n];
				rand_nums.push_back(random_float);
			}
			sort(rand_nums.begin(), rand_nums.end(), less<Float>());
			Float* prod = new Float[K];
			memset(prod, 0.0, sizeof(Float)*K);
			cache_mu_n(n);
			bool* inside_a = new bool[K];
			memset(inside_a, false, sizeof(bool)*K);
			
			for (PairVec::iterator it_a = act_alpha_n.begin(); it_a != act_alpha_n.end(); it_a++){
				Int k = it_a->first;
				alpha_n[k] = it_a->second;
				inside_a[k] = true;
			}
	
			for (vector<pair<Int, Float*>>::iterator it_b = act_beta[n].begin(); it_b != act_beta[n].end(); it_b++){
				Int offset = it_b->first;
				inside[offset] = true;
				beta_n[offset] = it_b->second;
			}
		
			vector<Float>::iterator current_index = rand_nums.begin();
			Float current_sum = 0.0;
			for (SparseVec::iterator it = ins->feature.begin(); it != ins->feature.end() && current_index != rand_nums.end(); current_index++, it++){
				current_sum += it->second;
				if (current_sum < (*current_index)){
					continue;
				}
				
				Float xij = 0.0;
				while (current_index != rand_nums.end() && current_sum >= (*current_index)){
					current_index++;
					xij += 1.0;
				}
				Int j = it->first;
				xij *= ((it->second > 0.0)?1:(-1));
				Float* wj = w[j];
				for (Int k = range_l; k < range_r; k++){
					prod[k] += wj[k]*xij;
				}
			}
			
			vector<Int>* ever_act_alpha_n = &(ever_act_alpha[n]);
			bool* is_pos_label_n = is_pos_label[n];
			bool* is_ever_active_n = is_ever_active[n];
			//need some msg here
			Float msg_L = 0.0, msg_R = 0.0;
			for (Int ii = range_l; ii < range_r; ii++){
				if (inside_a[ii]) continue;
				for (vector<Int>::iterator it_e = ever_act_alpha_n->begin(); it_e != ever_act_alpha_n->end(); it_e++){
					Int i = ii, j = *it_e;
					if (is_ever_active_n[i] && i >= j)
						continue;
					if (i > j){
						Int temp = i; i = j; j = temp;
					}
					Int offset = K*i + j;
					if (inside[offset]){
						msg_L = beta_n[offset][2] + beta_n[offset][3] - alpha_n[i] + mu_n[offset][F_LEFT];
						msg_R = beta_n[offset][1] + beta_n[offset][3] - alpha_n[j] + mu_n[offset][F_RIGHT];
					} else {
						Float beta_nij01, beta_nij10;
						recover_beta(n, i, j, alpha_n, mu_n, beta_nij10, beta_nij01);
						msg_L = beta_nij10 - alpha_n[i] + mu_n[offset][F_LEFT];
						msg_R = beta_nij01 - alpha_n[j] + mu_n[offset][F_RIGHT];
					}
					prod[i] -= eta * msg_L;
					prod[j] -= eta * msg_R;
				}
			}
			
			for (vector<pair<Int, Float*>>::iterator it_b = act_beta[n].begin(); it_b != act_beta[n].end(); it_b++){
				Int offset = it_b->first;
				Int i = offset / K, j = offset % K;
				assert(i < j);
				if (!is_ever_active_n[i] || !is_ever_active_n[j]){
					msg_L = beta_n[offset][2] + beta_n[offset][3] - alpha_n[i] + mu_n[offset][F_LEFT];
					msg_R = beta_n[offset][1] + beta_n[offset][3] - alpha_n[j] + mu_n[offset][F_RIGHT];
					prod[i] -= eta * msg_L;
					prod[j] -= eta * msg_R;
				}
			}

			Int max_k = -1;
			Float max_val = -INFI;
			for (Int k = range_l; k < range_r; k++){
				if (!inside_a[k] && prod[k] > max_val){
					max_k = k;
					max_val = prod[k];
				}
			}

			for (PairVec::iterator it = act_alpha_n.begin(); it != act_alpha_n.end(); it++){
				alpha_n[it->first] = 0.0;
			}

			if (max_val > (-1.0 * m) / (Float)(ins->feature.size()) ){
				act_alpha_n.push_back(make_pair(max_k, 0.0));
			}

			for (vector<pair<Int, Float*>>::iterator it_b = act_beta[n].begin(); it_b != act_beta[n].end(); it_b++){
				Int offset = it_b->first;
				inside[offset] = false;
				beta_n[offset] = NULL;
			}
			clean_mu_n(n);
			delete[] prod;
			delete[] inside_a;
		}

		void bi_update(Int n, Float** beta_new, Int iter){

			Float* Dk = new Float[4];
			Float* grad = new Float[4];

			//data
			vector<pair<Int, Float*>>* act_beta_n = &(act_beta[n]);

			for (PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
				alpha_n[it_alpha->first] = it_alpha->second;
			}
			cache_mu_n(n);
			//indicator of ground truth labels
			bool* is_pos_label_n = is_pos_label[n];

			Float Qii = (1.0+eta*2);
			for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n->begin(); it_beta != act_beta_n->end(); it_beta++){
					Int offset = it_beta->first;
					Float* beta_nij = it_beta->second;
					Int i = offset / K, j = offset % K;
					assert(i < j);
					Float* mu_nij = mu_n[offset];
					//compute gradient
					for(int d=0;d<4;d++)
						grad[d] = 0.0;

					grad[3] += v[i][j];
					assert(alpha_n != NULL);
					assert(mu_nij != NULL);
					assert(beta_nij != NULL);
					Float msg_L = beta_nij[2]+beta_nij[3] - alpha_n[i] + mu_nij[F_LEFT];
					grad[2] += eta*( msg_L ); //2:10
					grad[3] += eta*( msg_L ); //3:11

					Float msg_R = beta_nij[1]+beta_nij[3] - alpha_n[j] + mu_nij[F_RIGHT];
					grad[1] += eta*( msg_R ); //1:01
					grad[3] += eta*( msg_R ); //1:11
					//compute max gradient among negative bits
					Int max_neg_bits = -1;
					Float max_val_neg = -INFI;
					Int pos_bits = 2*is_pos_label_n[i] + is_pos_label_n[j];
					for(Int d=0;d<4;d++){
						if( d != pos_bits ){
							if (grad[d] > max_val_neg){
								max_neg_bits = d;
								max_val_neg = grad[d];
							}
						}
					}
					for (Int d = 0; d < 4; d++)
						beta_new[offset][d] = 0.0;

					if (grad[pos_bits] - max_val_neg < 0){
						beta_new[offset][pos_bits] = C;
						beta_new[offset][max_neg_bits] = -C;
					}
					Float delta_beta11 = beta_new[offset][3] - beta_nij[3];
					Float delta_beta10 = beta_new[offset][2] - beta_nij[2];
					Float delta_beta01 = beta_new[offset][1] - beta_nij[1];
					Float up = - (delta_beta11 * v[i][j] + eta * msg_L * (delta_beta11 + delta_beta10) + eta * msg_R * (delta_beta11 + delta_beta01) );
					Float down = delta_beta11 * delta_beta11 + eta * ((delta_beta10 + delta_beta11) * (delta_beta10 + delta_beta11) + (delta_beta01 + delta_beta11) * (delta_beta01 + delta_beta11));
					Float gamma = 0.0;
					if (fabs(down) > 1e-12) gamma = up / down;
					if (gamma < 0.0) gamma = 0.0;
					if (gamma > 1.0) gamma = 1.0;

					//record beta new values
					for (int d=0;d<4;d++)
						beta_new[offset][d] = beta_nij[d] + gamma * (beta_new[offset][d] - beta_nij[d]);
					
			}
			
			for (PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
				alpha_n[it_alpha->first] = 0.0;
			}
			
			clean_mu_n(n);
			delete[] Dk;
			delete[] grad;
		}

		void bi_subSolve(Int n, Float** beta_new){

			Float* Dk = new Float[4];
			Float* grad = new Float[4];

			//data
			vector<pair<Int, Float*>>* act_beta_n = &(act_beta[n]);
			//memset(alpha_n, 0.0, sizeof(Float)*K);
			for (PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
				alpha_n[it_alpha->first] = it_alpha->second;
			}
			cache_mu_n(n);
			//indicator of ground truth labels
			bool* is_pos_label_n = is_pos_label[n];

		
			Float Qii = (1.0+eta*2);
			for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n->begin(); it_beta != act_beta_n->end(); it_beta++){
					Int offset = it_beta->first;
					Float* beta_nij = it_beta->second;
					Int i = offset / K, j = offset % K;
					assert(i < j);
					Float* mu_nij = mu_n[offset];
					//compute gradient
					for(int d=0;d<4;d++)
						grad[d] = - Qii*beta_nij[d];

					grad[3] += v[i][j];
					assert(alpha_n != NULL);
					assert(mu_nij != NULL);
					assert(beta_nij != NULL);
					Float msg_L = beta_nij[2]+beta_nij[3] - alpha_n[i] + mu_nij[F_LEFT];
					grad[2] += eta*( msg_L ); //2:10
					grad[3] += eta*( msg_L ); //3:11

					Float msg_R = beta_nij[1]+beta_nij[3] - alpha_n[j] + mu_nij[F_RIGHT];
					grad[1] += eta*( msg_R ); //1:01
					grad[3] += eta*( msg_R ); //1:11
					
					//compute Dk
					int d_nij = 2*is_pos_label_n[i] + is_pos_label_n[j];
					for(Int d=0;d<4;d++){
						if( d != d_nij )
							Dk[d] = grad[d];
						else
							Dk[d] = grad[d] + Qii*C;
					}
					
					//sort according to D_k
					sort( Dk, Dk+4,  greater<Float>() );
					//compute b by traversing D_k in descending order
					Float b = Dk[0] - Qii*C;
					Int r;
					for( r=1; r<4 && b<r*Dk[r]; r++)
						b += Dk[r];
					b = b / r;
					
					//record beta new values
					for (int d=0;d<4;d++)
						beta_new[offset][d] = min( (Float)((d!=d_nij)?0.0:C), (b-grad[d])/Qii );
					
			}
			
			for (PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
				alpha_n[it_alpha->first] = 0.0;
			}
			
			clean_mu_n(n);
			delete[] Dk;
			delete[] grad;
		}
		
		void bi_brute_force_search(Int n, vector<pair<Int, Float*>>& act_beta_n){

			Int max_k1k2 = -1;
			Float max_val = -1e300;	
			
			for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n.begin(); it_beta != act_beta_n.end(); it_beta++){
				Int offset = it_beta->first;
				inside[offset] = true;
			}
			
			Instance* ins = data->at(n);
			Labels* pos_labels = &(ins->labels);
			PairVec* act_alpha_n = &(act_alpha[n]);
			vector<Int>* ever_act_alpha_n = &(ever_act_alpha[n]);
			cache_mu_n(n);

			bool* is_pos_label_n = is_pos_label[n];

			//memset(alpha_n, 0.0, sizeof(Float)*K);
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				alpha_n[it_alpha->first] = it_alpha->second;
			}

			//area 1; y^n_i = y^n_j = 1
			for (Int i = 0; i < K; i++){
				Int Ki = K * i;
				for (Int j = i+1; j < K; j++){
					Float alpha_ni = alpha_n[i];
					Float alpha_nj = alpha_n[j];
					Int offset = Ki + j;
					if (inside[offset])
						continue;
					
					Float beta_nij01, beta_nij10;
					recover_beta(n, i, j, alpha_n, mu_n, beta_nij10, beta_nij01);
					Float msg_L = beta_nij10 - alpha_n[i] + mu_n[offset][F_LEFT];
					Float msg_R = beta_nij01 - alpha_n[j] + mu_n[offset][F_RIGHT];

					Float val = v[i][j] + eta*(msg_L + msg_R);
					if (val > max_val){
						max_k1k2 = offset;
						max_val = val;
					}
					if (is_pos_label_n[i]){
						//beta^{10} is positive, thus in range [0, C], should consider -gradient
						msg_L *= (-1);
					}
					if (is_pos_label_n[j]){
						//beta^{01} is positive, thus in range [0, C], should consider -gradient
						msg_R *= (-1);
					}
					if (eta * msg_L > max_val){
						max_k1k2 = offset;
						max_val = eta * msg_L;
					}
					if (eta * msg_R > max_val){
						max_k1k2 = offset;
						max_val = eta * msg_R;
					}
				}
			}

			for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n.begin(); it_beta != act_beta_n.end(); it_beta++){
				Int offset = it_beta->first;
				inside[offset] = false;
			}
			
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				alpha_n[it_alpha->first] = 0.0;
			}

			if (max_val > 0.0){
				Float* temp_float = new Float[4];
				memset(temp_float, 0.0, sizeof(Float)*4);
				act_beta_n.push_back(make_pair(max_k1k2, temp_float));
			}
			clean_mu_n(n);
			
			assert(act_beta_n.size() <= K*(K-1)/2);
		}

		void bi_search(Int n, vector<pair<Int, Float*>>& act_beta_n){

			Int max_k1k2 = -1;
			Float max_val = -1e300;	
			
			for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n.begin(); it_beta != act_beta_n.end(); it_beta++){
				Int offset = it_beta->first;
				inside[offset] = true;
			}
			
			Instance* ins = data->at(n);
			Labels* pos_labels = &(ins->labels);
			PairVec* act_alpha_n = &(act_alpha[n]);
			vector<Int>* ever_act_alpha_n = &(ever_act_alpha[n]);
			cache_mu_n(n);

			bool* is_pos_label_n = is_pos_label[n];

			//memset(alpha_n, 0.0, sizeof(Float)*K);
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				alpha_n[it_alpha->first] = it_alpha->second;
			}

			//area 1; y^n_i = y^n_j = 1
			for (vector<Int>::iterator it_alpha = ever_act_alpha_n->begin(); it_alpha != ever_act_alpha_n->end(); it_alpha++){
				Int ii = *it_alpha;
				for (vector<Int>::iterator it_alpha2 = it_alpha+1; it_alpha2 != ever_act_alpha_n->end(); it_alpha2++){
					Int i = ii;
					Int j = *it_alpha2;
					if (i > j){
						Int temp = i; i = j; j = temp;
					}
					Float alpha_ni = alpha_n[i];
					Int Ki = K*i;
					Float alpha_nj = alpha_n[j];
					Int offset = Ki + j;
					if (inside[offset])
						continue;
					
					Float beta_nij01, beta_nij10;
					recover_beta(n, i, j, alpha_n, mu_n, beta_nij10, beta_nij01);
					Float msg_L = beta_nij10 - alpha_n[i] + mu_n[offset][F_LEFT];
					Float msg_R = beta_nij01 - alpha_n[j] + mu_n[offset][F_RIGHT];

					Float val = v[i][j] + eta*(msg_L + msg_R);
					if (val > max_val){
						max_k1k2 = offset;
						max_val = val;
					}
					if (is_pos_label_n[i]){
						//beta^{10} is positive, thus in range [0, C], should consider -gradient
						msg_L *= (-1);
					}
					if (is_pos_label_n[j]){
						//beta^{01} is positive, thus in range [0, C], should consider -gradient
						msg_R *= (-1);
					}
					if (eta * msg_L > max_val){
						max_k1k2 = offset;
						max_val = eta * msg_L;
					}
					if (eta * msg_R > max_val){
						max_k1k2 = offset;
						max_val = eta * msg_R;
					}
				}
			}
		
			bool* is_ever_active_n = is_ever_active[n];	
			search_matrix(v_heap, is_ever_active_n, max_val, max_k1k2, v_heap_size, inside, K);

			for (vector<pair<Int, Float*>>::iterator it_beta = act_beta_n.begin(); it_beta != act_beta_n.end(); it_beta++){
				Int offset = it_beta->first;
				inside[offset] = false;
			}
			
			for (PairVec::iterator it_alpha = act_alpha_n->begin(); it_alpha != act_alpha_n->end(); it_alpha++){
				alpha_n[it_alpha->first] = 0.0;
			}

			if (max_val > 0.0){
				Float* temp_float = new Float[4];
				memset(temp_float, 0.0, sizeof(Float)*4);
				act_beta_n.push_back(make_pair(max_k1k2, temp_float));
			}
			clean_mu_n(n);
			
			assert(act_beta_n.size() <= K*(K-1)/2);
		}

		int mat_bottom = 0, mat_top = 0;
		void search_matrix(ArrayHeap heap, bool* is_ever_active_n, Float& max_val, Int& max_k1k2, Int heap_size, bool* inside, Int K){
			vector<Int> q;
			q.push_back(0);
			mat_bottom++;
			while (!q.empty()){
				mat_top++;
				Int index = q.back();
				q.pop_back();
				pair<Float, Int> p = heap[index];
				Int offset = p.second;
				Int i = offset / K;
				Int j = offset % K;
				assert(i < j);
				if (v[i][j] <= max_val){
					continue;
				}
				if (inside[offset] || (is_ever_active_n[i] && is_ever_active_n[j])){
					if (index*2+1 < heap_size){
						q.push_back(index*2+1);
						if (index*2+2 < heap_size){
							q.push_back(index*2+2);
						}
					}
					continue;
				}
				//must have v[i][j] > max_val
				max_val = v[i][j];
				max_k1k2 = offset;
			}
		}

		inline void recover_beta(Int n, Int i, Int j, Float* alpha_n, Float** mu_n, Float& beta_nij10, Float& beta_nij01){
			beta_nij10 = 0.0; beta_nij01 = 0.0;
			Int offset = K*i+j;
			Float* mu_nij = mu_n[offset];
			//compute gradient
			Float Qii = (1.0+eta*2);
			for(int d=0;d<3;d++)
				grad[d] = 0.0;
			
			//Float beta_nij_left1_sum = beta_n[Ki+j][2]+beta_n[Ki+j][3];
			Float msg_L = - alpha_n[i] + mu_nij[F_LEFT];
			grad[2] += eta*( msg_L ); //2:10

			Float msg_R = - alpha_n[j] + mu_nij[F_RIGHT];
			grad[1] += eta*( msg_R ); //1:01

			//compute Dk
			Float* Dk = new Float[3];
			int d_nij = 2*is_pos_label[n][i] + is_pos_label[n][j];
			Dk[0] = grad[0]; Dk[1] = grad[1]; Dk[2] = grad[2];
			Dk[d_nij] += Qii*C;

			//sort according to D_k
			sort( Dk, Dk+3,  greater<Float>() );

			//compute b by traversing D_k in descending order
			Float b = Dk[0] - Qii*C;
			Int r;
			for( r=1; r<3 && b<r*Dk[r]; r++)
				b += Dk[r];
			b = b / r;

			//record beta new values
			for (int d=0;d<3;d++){
				grad[d] = 0.0;
			}
			beta_nij10 = min( (Float)((d_nij!=2)?0.0:C), (b-grad[2])/Qii );
			beta_nij01 = min( (Float)((d_nij!=1)?0.0:C), (b-grad[1])/Qii );
			delete[] Dk;

///////////////////
//DIFFERENT ALGO
			/*Int offset = K*i + j;
			Float tmp_L = alpha_n[i] - mu[n][offset][F_LEFT];
			Float tmp_R = alpha_n[j] - mu[n][offset][F_RIGHT];

			//shouldn't have both positive here
			if (is_pos_label[n][i]){
				tmp_L -= C;
				beta_nij10 = C;
			}
			if (is_pos_label[n][j]){
				tmp_R -= C;
				beta_nij01 = C;
			}
			//solve problem: 
			//   min (beta^{10} - tmp_L)^2 + (beta^{01} - tmp_R)^2
			//   s.t. beta^{01}, beta^{10} \in [-C, 0], beta^{01} + beta^{10} \in [-C, 0]
			proj(tmp_L, tmp_R);
			beta_nij10 += tmp_L;
			beta_nij01 += tmp_R;*/	
		}

		inline void cache_mu_n(Int n){
			for (vector<pair<Int, Float*>>::iterator it_m = mu[n].begin(); it_m != mu[n].end(); it_m++){
				Int offset = it_m->first;
				mu_n[offset][0] = it_m->second[0];
				mu_n[offset][1] = it_m->second[1];
			}
		}
		
		inline void clean_mu_n(Int n){
			for (vector<pair<Int, Float*>>::iterator it_m = mu[n].begin(); it_m != mu[n].end(); it_m++){
				Int offset = it_m->first;
				mu_n[offset][0] = 0.0;
				mu_n[offset][1] = 0.0;
			}
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
			
			Int Ns = N/50;

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
		
		Float heldout_acc_joint(){
			
			vector<Instance*>* heldout_data = &(heldout_prob->data);
			Int Ns = heldout_data->size() / 10;
			Int hit=0;
			Int n = 0;
			Int* pred = new Int[K];

			for (vector<Instance*>::iterator it_heldout = heldout_data->begin(); it_heldout != heldout_data->end(); it_heldout++){
				if ((n++) >= Ns)
					break;
				Instance* ins = *it_heldout;	
				model->LPpredict(ins, pred);
				for(Int k=0;k<K;k++){
					int yk = (find(ins->labels.begin(), ins->labels.end(), k) != ins->labels.end())? 1:0;
					if( pred[k] == yk )
						hit++;
				}
				if( n % 10 == 0 )
					cerr << "." ;
			}

			delete[] pred;
			
			return (Float)hit/Ns/K;
		}

		Float dual_obj(){

			Float u_obj = 0.0;
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
				bool* is_pos_label_n = is_pos_label[n];
				vector<Int>* ever_act_alpha_n = &(ever_act_alpha[n]);
				cache_mu_n(n);
				//memset(alpha_n, 0.0, sizeof(Float)*K);
				for (PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
					alpha_n[it_alpha->first] = it_alpha->second;
				}
				for (vector<pair<Int, Float*>>::iterator it_b = act_beta[n].begin(); it_b != act_beta[n].end(); it_b++){
					Int offset = it_b->first;
					inside[offset] = true;
					beta_n[offset] = it_b->second;
				}
				// adds act_beta
				for(vector<Int>::iterator it_ever = ever_act_alpha_n->begin(); it_ever != ever_act_alpha_n->end(); it_ever++){
					Int ii = *it_ever;
					for(vector<Int>::iterator it_ever2 = it_ever+1; it_ever2 != ever_act_alpha_n->end(); it_ever2++){
						Int i = ii;
						Int j = *it_ever2;
						if (i > j){
							Int temp = i; i = j; j = temp;
						}
						Int offset = K*i+j;
						if (inside[offset]){
							Float delta_mu_l = beta_n[offset][2] + beta_n[offset][3] - alpha_n[i] + mu_n[offset][0];
							Float delta_mu_r = beta_n[offset][1] + beta_n[offset][3] - alpha_n[j] + mu_n[offset][1];
							p_inf += delta_mu_l * delta_mu_l;
							p_inf += delta_mu_r * delta_mu_r;
						} else {
							
							Float tmp_L = alpha_n[i] - mu_n[offset][F_LEFT];
							Float tmp_R = alpha_n[j] - mu_n[offset][F_RIGHT];
							Float beta_nij01 = 0.0, beta_nij10 = 0.0;

							//shouldn't have both positive here
							if (is_pos_label_n[i]){
								tmp_L -= C;
								beta_nij10 = C;
							}
							if (is_pos_label_n[j]){
								tmp_R -= C;
								beta_nij01 = C;
							}
							//solve problem: 
							//   min (beta^{10} - tmp_L)^2 + (beta^{01} - tmp_R)^2
							//   s.t. beta^{01}, beta^{10} \in [-C, 0], beta^{01} + beta^{10} \in [-C, 0]
							proj(tmp_L, tmp_R);
							beta_nij10 += tmp_L;
							beta_nij01 += tmp_R;
							Float delta_mu_l = beta_nij10 - alpha_n[i] + mu_n[offset][0];
							Float delta_mu_r = beta_nij01 - alpha_n[j] + mu_n[offset][1];
							p_inf += delta_mu_l * delta_mu_l;
							p_inf += delta_mu_r * delta_mu_r;
						}
					}
				}
				bool* is_ever_active_n = is_ever_active[n];
				for (vector<pair<Int, Float*>>::iterator it_beta = act_beta[n].begin(); it_beta != act_beta[n].end(); it_beta++){
					Int offset = it_beta->first;
					Int i = offset / K, j = offset % K;
					if (is_ever_active_n[i] && is_ever_active_n[j]){
						beta_n[offset] = NULL;
						inside[offset] = false;
						continue;
					}
					//assert(!is_ever_active_n[i] && !is_ever_active_n[j]);
					Float* beta_nij = it_beta->second;
					Float delta_mu_l = beta_nij[2] + beta_nij[3] - alpha_n[i] + mu_n[offset][0];
					Float delta_mu_r = beta_nij[1] + beta_nij[3] - alpha_n[j] + mu_n[offset][1];
					p_inf += delta_mu_l * delta_mu_l;
					p_inf += delta_mu_r * delta_mu_r;
					beta_n[offset] = NULL;
					inside[offset] = false;
				}

				for (PairVec::iterator it_alpha = act_alpha[n].begin(); it_alpha != act_alpha[n].end(); it_alpha++){
					alpha_n[it_alpha->first] = 0.0;
				}
				clean_mu_n(n);
			}
			p_inf *= eta/2.0;
	
			return u_obj + bi_obj + p_inf;
		}
		
		// project (a,b) onto simplex {(x,y) | x \in [-C, 0], y \in [-C, 0], x+y \in [-C, 0] }
		inline void proj(Float& a, Float& b){
			if (a <= 0.0 && b <= 0.0){
				if (a + b <= -C){
					Float t = 0.5*(a + b + C);
					a -= t;
					b -= t;
				} else {
					//otherwise (a, b) is already in simplex, no need for projection
					return;
				}
			}
			if (a > 0.0){
				a = 0.0;
			} else {
				if (a < -C)
					a = -C;
			}
			if (b > 0.0){
				b = 0.0;
			} else {
				if (b < -C)
					b = -C;
			}
		}

		MultilabelProblem* prob;

		//for heldout option
		MultilabelProblem* heldout_prob;		
		Int early_terminate;
		Int heldout_period;

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
		Float* alpha_n;
		Float** beta_n;
		Float** mu_n;

		Float* Q_diag;
		PairVec* act_alpha; //N*K dual variables for unigram factor
		vector<Int>* ever_act_alpha;	// maintain alpha indices that are ever active (necessary for non-zero messages)
		bool** is_ever_active; // N*K bool arrayis ever active

		vector<pair<Int, Float*>>* act_beta;
		//Float*** beta; //N*M*4 dual variables for bigram factor (allocate N*K^2 instead of N*M for convenience)

		Float** w; //D*K primal variables for unigram factor
		Float** v; //K^2 primal variables for bigram factor

		ArrayHeap v_heap;
		Int v_heap_size;	
		Int* v_index;

		//uni_search
		Int split_up_rate;
		Int speed_up_rate;

		//bi_search
		bool using_brute_force_search;
		
		//l1 norm of each feature vector
		Float* l1_norm;

		bool do_subSolve;		

		vector<pair<Int, Float*>>* mu; // N*K^2 * 2 Lagrangian Multipliers on consistency constraInts

		Int max_iter;
		Float eta;
		Float admm_step_size;
		
		Model* model;
};
