#include "util.h"
#include "multilabel.h"

class BDMMsolve{

	public:
		enum Direction {F_LEFT=0, F_RIGHT=1, NUM_DIRECT};

		BDMMsolve(Param* param){

			//Parse info from ChainProblem
			prob = param->prob;
			data = &(prob->data);
			N = data->size();
			D = prob->D;
			K = prob->K;

			C = param->C;
			eta = param->eta;
			max_iter = param->max_iter;
			admm_step_size = param->admm_step_size;

			//allocate dual variables
			alpha = new Float*[N];
			for(Int i=0;i<N;i++){
				alpha[i] = new Float[K];
				for(Int k=0;k<K;k++)
					alpha[i][k] = 0.0;
			}
			beta = new Float**[N];
			for(Int n=0;n<N;n++){
				beta[n] = new Float*[K*K];
				for(Int kk=0;kk<K*K;kk++)
					beta[n][kk] = NULL;
				for(Int i=0;i<K;i++)
					for(Int j=i+1;j<K;j++){
						beta[n][i*K+j] = new Float[4];
						memset( beta[n][i*K+j], 0.0, sizeof(Float)*4 );
					}
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
		}

		~BDMMsolve(){

			//dual variables
			for(Int i=0;i<N;i++)
				delete[] alpha[i];
			delete[] alpha;
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
			
			Float p_inf;
			for(Int iter=0;iter<max_iter;iter++){

				random_shuffle(sample_index, sample_index+N);
				//update unigram dual variables
				for(Int r=0;r<N;r++){

					Int n = sample_index[r];

					//subproblem solving
					uni_subSolve(n, alpha_new);

					//maIntain relationship between w and alpha
					Float* alpha_n = alpha[n];
					Instance* ins = data->at(n);
					for(SparseVec::iterator it=ins->feature.begin(); it!=ins->feature.end(); it++){
						Int j = it->first;
						Float fval = it->second;
						Float* wj = w[j];
						for(Int k=0;k<K;k++){
							wj[k] += fval * (alpha_new[k]-alpha_n[k]);
						}
					}

					//update alpha
					for(Int k=0;k<K;k++){
						alpha_n[k] = alpha_new[k];
					}
				}
				
				//update bigram dual variables	
				random_shuffle(sample_index, sample_index+N);

				for(Int r=0;r<N;r++){
					Int n = sample_index[r];
					//subproblem solving
					bi_subSolve(n, beta_new);

					//maIntain relationship between v and beta
					Float** beta_n = beta[n];
					for(Int i=0;i<K;i++){
						Int Ki = K*i;
						for(Int j=i+1;j<K;j++)
							v[i][j] += beta_new[Ki+j][3] - beta_n[Ki+j][3];
					}
					
					//update beta
					for(Int i=0;i<K;i++){
						Int Ki = K*i;
						for(Int j=i+1;j<K;j++){
							for(int y1y2=0;y1y2<4;y1y2++)
								beta_n[Ki+j][y1y2] = beta_new[Ki+j][y1y2];
						}
					}
				}

				//ADMM update (enforcing consistency)
				Float* mu_ij;
				Float p_inf_nij;
				p_inf = 0.0;
				for(Int n=0;n<N;n++){
					Instance* ins = data->at(n);
					Float** beta_n = beta[n];
					Float* alpha_n = alpha[n];
					for(Int i=0;i<K;i++){
						Int Ki = K*i;
						for(Int j=i+1;j<K;j++){
							Float* mu_ij = mu[n][Ki+j];
							//left
							p_inf_nij = beta_n[Ki+j][2]+beta_n[Ki+j][3]-alpha_n[i];
							mu_ij[F_LEFT] += admm_step_size*p_inf_nij;
							p_inf += p_inf_nij * p_inf_nij;
							//right
							p_inf_nij = beta_n[Ki+j][1]+beta_n[Ki+j][3]-alpha_n[j];
							mu_ij[F_RIGHT] += admm_step_size*p_inf_nij;
							p_inf += p_inf_nij * p_inf_nij;
						}
					}
				}
				p_inf *= (eta/2);

				double beta_nnz=0.0;
				for(Int n=0;n<N;n++){
					for(Int i=0;i<K;i++){
						Int Ki = K*i;
						for(Int j=i+1; j<K; j++){
							if( fabs(beta[n][Ki+j][3]) > 1e-6 )
								beta_nnz+=1.0;
						}
					}
				}
				beta_nnz /= N;

				double alpha_nnz=0.0;
				for(Int n=0;n<N;n++){
					for(Int k=0;k<K;k++){
						if( fabs(alpha[n][k]) > 1e-6 )
							alpha_nnz += 1.0;
					}
				}
				alpha_nnz/=N;
				cerr << "i=" << iter << ", a_nnz=" << alpha_nnz << ", b_nnz=" << beta_nnz 
					<< ", infea=" << p_inf <<  ", d_obj=" << dual_obj() << ", uAcc=" << train_acc_unigram() << endl;
				//if(iter%10==9)
				//	cerr << ", Acc=" << train_acc_joint() << endl;
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

			Instance* ins = data->at(n);
			Float Qii = Q_diag[n];
			Float* alpha_n = alpha[n];
			Float** beta_n = beta[n];
			Float** mu_n = mu[n];
			
			bool* is_pos_label = new bool[K];
			for(Int k=0;k<K;k++)
				is_pos_label[k] = false;
			for(int i=0;i<ins->labels.size();i++)
				is_pos_label[ ins->labels[i] ] = true;
			
			Float* grad = new Float[K];
			memset(grad, 0.0, sizeof(Float)*K);
			for(SparseVec::iterator it=ins->feature.begin(); it!=ins->feature.end(); it++){
				Float* wj = w[it->first];
				for(Int k=0;k<K;k++){
					grad[k] += wj[k]*it->second;
				}
			}
			for(Int k=0;k<K;k++){
				if( !is_pos_label[k] )
					grad[k] += 1.0;
				else
					grad[k] += -1.0;
			}
			
			//messages
			for(Int i=0;i<K;i++){
				Int Ki = K*i;
				for(Int j=i+1;j<K;j++){//from each bigram
					Float* beta_nij = beta_n[Ki+j];
					//bigram to left
					grad[i] -= eta*( beta_nij[2] + beta_nij[3] - alpha_n[i] + mu_n[Ki+j][F_LEFT] );
					//bigram to right
					grad[j] -= eta*( beta_nij[1] + beta_nij[3] - alpha_n[j] + mu_n[Ki+j][F_RIGHT] );
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

			for(Int k=0;k<K;k++)
				alpha_new[k] = min( max( alpha_n[k] - grad[k]/Qii, L[k] ), U[k] );
			
			delete[] is_pos_label;
			delete[] grad;
			delete[] U;
			delete[] L;
		}

		void bi_subSolve(Int n, Float** beta_new){

			Int Ksq = K*K;
			Float* grad = new Float[Ksq*4];
			Float* Dk = new Float[Ksq*4];

			//data
			Instance* ins = data->at(n);
			Float** beta_n = beta[n];
			Float* alpha_n = alpha[n];
			Float** mu_n = mu[n];
			//indicator of ground truth labels
			bool* is_pos_label = new bool[K];
			for(Int k=0;k<K;k++)
				is_pos_label[k] = false;
			for(vector<int>::iterator it=ins->labels.begin(); it!=ins->labels.end(); it++)
				is_pos_label[*it] = true;
			
			Float Qii = (1.0+eta*2);
			for(Int i=0;i<K;i++){
				Int Ki = K*i;

				for(Int j=i+1; j<K; j++){

					Int offset = (Ki+j)*4;
					//compute gradient
					for(int d=0;d<4;d++)
						grad[offset+d] = - Qii*beta_n[Ki+j][d];
					
					grad[offset + 3] += v[i][j];

					Float beta_nij_left1_sum = beta_n[Ki+j][2]+beta_n[Ki+j][3];
					grad[offset + 2] += eta*( beta_nij_left1_sum - alpha_n[i] + mu_n[Ki+j][F_LEFT] ); //2:10
					grad[offset + 3] += eta*( beta_nij_left1_sum - alpha_n[i] + mu_n[Ki+j][F_LEFT] ); //3:11

					Float beta_nij_right1_sum = beta_n[Ki+j][1]+beta_n[Ki+j][3];
					grad[offset + 1] += eta*( beta_nij_right1_sum - alpha_n[j] + mu_n[Ki+j][F_RIGHT] ); //1:01
					grad[offset + 3] += eta*( beta_nij_right1_sum - alpha_n[j] + mu_n[Ki+j][F_RIGHT] ); //1:11
					
					//compute Dk
					int d_nij = 2*is_pos_label[i] + is_pos_label[j];
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
					for(int d=0;d<4;d++)	
						beta_new[Ki+j][d] = min( (Float)((d!=d_nij)?0.0:C), (b-grad[offset+d])/Qii );
				}
			}
			

			delete[] is_pos_label;
			delete[] grad;
			delete[] Dk;
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

			Float hit=0;
			Float* pred = new Float[K];
			for(Int n=0;n<Ns;n++){
				Instance* ins = data->at(n);
				model->LPpredict(ins, pred);
				for(Int k=0;k<K;k++){
					Float yk = (find(ins->labels.begin(), ins->labels.end(), k) != ins->labels.end())? 1:0;
					//if( pred[k] == yk )
					hit+= 1 - fabs(pred[k] - yk);
				}
				
				if( n%10 ==0 )
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
				for(Int k=0;k<K;k++){
					if( find(ins->labels.begin(), ins->labels.end(), k) == ins->labels.end() )
						u_obj += alpha[n][k];
					else
						u_obj += - alpha[n][k];
				}
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
				Float* alpha_n = alpha[n];
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
			
			return u_obj + bi_obj + p_inf;
		}

		MultilabelProblem* prob;

		vector<Instance*>* data;
		Float C;
		Int N;
		Int D;
		Int K;

		Float* Q_diag;
		Float** alpha; //N*K dual variables for unigram factor
		Float*** beta; //N*M*4 dual variables for bigram factor (allocate N*K^2 instead of N*M for convenience)

		Float** w; //D*K primal variables for unigram factor
		Float** v; //K^2 primal variables for bigram factor

		Float*** mu; // N*K^2 * 2 Lagrangian Multipliers on consistency constraInts

		Int max_iter;
		Float eta;
		Float admm_step_size;
		
		Model* model;
};
