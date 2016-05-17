#include "multilabel.h"
#include "BDMMsolve.h"
#include "BCFWsolve.h"

double overall_time = 0.0;

void exit_with_help(){
	cerr << "Usage: ./train (options) [train_data] (model)" << endl;
	cerr << "options:" << endl;
	cerr << "-s solver: (default 0)" << endl;
	cerr << "	0 -- BDMM" << endl;
	//cerr << "	1 -- Active Block Coordinate Descent" << endl;
	cerr << "	1 -- BCFW" << endl;
	//cerr << "-l lambda: L1 regularization weight (default 1.0)" << endl;
	cerr << "-c cost: cost of each sample (default 1)" << endl;
	cerr << "-r speed_up_rate: using 1/r fraction of samples (default min(max(DK/(log(K)nnz(X)),1),d/5) )" << endl;
	cerr << "-q split_up_rate: choose 1/q fraction of [K]" << endl;
	cerr << "-m max_iter: maximum number of iterations allowed (default 20)" << endl;
	//cerr << "-i im_sampling: Importance sampling instead of uniform (default not)" << endl;
	//cerr << "-g max_select: maximum number of greedy-selected dual variables per sample (default 1)" << endl;
	//cerr << "-p post_train_iter: #iter of post-training w/o L1R (default 0)" << endl;
	cerr << "-h heldout data set: use specified heldout data set" << endl;
	cerr << "-b brute_force search: use naive search (default off)" << endl;
	//cerr << "-w write_model_period: write model file every (arg) iterations (default max_iter)" << endl;
	cerr << "-t eta: set eta to (arg)" << endl;
	cerr << "-o heldout_period: period(#iters) to report heldout accuracy (default 10)" << endl;
	cerr << "-e early_terminate: stop if heldout accuracy doesn't increase in (arg) iterations (need -h) (default 3)" << endl;
	cerr << "-a admm_step_size: admm update step size (default 1.0) " << endl;
	cerr << "-u non-fully corrective update: use non-fully corrective update (default off) " << endl;
	exit(0);
}

void parse_cmd_line(int argc, char** argv, Param* param){

	int i;
	for(i=1;i<argc;i++){
		if( argv[i][0] != '-' )
			break;
		if( ++i >= argc )
			exit_with_help();

		switch(argv[i-1][1]){
			
			case 's': param->solver = atoi(argv[i]);
				  break;
//			case 'l': param->lambda = atof(argv[i]);
//				  break;
			case 'c': param->C = atof(argv[i]);
				  break;
			case 'm': param->max_iter = atoi(argv[i]);
				  break;
//			case 'g': param->max_select = atoi(argv[i]);
//				  break;
			case 'r': param->speed_up_rate = atoi(argv[i]);
				  break;
//			case 'i': param->using_importance_sampling = true; --i;
//				  break;
			case 'q': param->split_up_rate = atoi(argv[i]);
				  break;
//			case 'p': param->post_solve_iter = atoi(argv[i]);
//				  break;
			case 'h': param->heldoutFname = argv[i];
				  break;
			case 'b': param->using_brute_force_search = true; --i;
				  break;
//			case 'w': param->write_model_period = atoi(argv[i]);
//				  break;
			case 'e': param->early_terminate = atoi(argv[i]);
				  break;
			case 'a': param->admm_step_size = atof(argv[i]);
				  break;
			case 't': param->eta = atof(argv[i]);
				  break;
			case 'o': param->heldout_period = atoi(argv[i]);
				  break;
			case 'u': param->do_subSolve = false; --i;
			          break;
			default:
				  cerr << "unknown option: -" << argv[i-1][1] << endl;
				  exit(0);
		}
	}

	if(i>=argc)
		exit_with_help();

	param->trainFname = argv[i];
	i++;
	if( i<argc )
		param->modelFname = argv[i];
	else{
		param->modelFname = new char[FNAME_LEN];
		strcpy(param->modelFname,"model");
	}
}

void writeModel( char* fname, Model* model ){

    ofstream fout(fname);
    int K = model->K, D = model->D;
    Float** w = model->w;
    Float** v = model->v;
    vector<string>* label_name_list = model->label_name_list;
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

int main(int argc, char** argv){

	Param* param = new Param();
	parse_cmd_line(argc, argv, param);
	
	param->prob = new MultilabelProblem(param->trainFname);

	if (param->heldoutFname != NULL){
		cerr << "using heldout data set: " << param->heldoutFname << endl;
		param->heldout_prob = new MultilabelProblem(param->heldoutFname);
	}

	cerr << "D=" << param->prob->D << endl;
	cerr << "K=" << param->prob->K << endl;
	cerr << "N=" << param->prob->N << endl;

	overall_time = -omp_get_wtime();
	/*int m = 10;
	int me = 0;
	int nf = 0;
	int n = 10;
	SparseVec* A = new SparseVec[m+me];
	SparseVec* At = new SparseVec[n+nf];
	transpose(A, m+me, n+nf, At);
	double* b = new double[m+me];
	double* c = new double[n+nf];
	double* x = new double[n+nf];
	double* w = new double[m+me];
	
	LPsolve(n,nf,m,me, A, At, b, c, x, w);
	*/
	if (param->solver == 0){
		BDMMsolve* solver = new BDMMsolve(param);
		Model* model = solver->solve();
	} else {
		BCFWsolve* solver = new BCFWsolve(param);
		Model* model = solver->solve();
        writeModel(param->modelFname, model);
	}

	overall_time += omp_get_wtime();
	cerr << "overall time=" << overall_time << endl;
	
	return 0;
}
