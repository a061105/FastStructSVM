#include "multilabel.h"
#include "BDMMsolve.h"
//#include "BCFWsolve.h"

/*void writeModel(Model* model, Param* param){
	

}*/

int main(int argc, char** argv){
	
	if( argc < 1+1 ){
		cerr << "./train [data] (model)" << endl;
		exit(0);
	}

	Param* param = new Param();
	param->trainFname = argv[1];
	if( argc > 1+1 )
		param->modelFname = argv[2];
	else
		param->modelFname = "model";
	
	param->prob = new MultilabelProblem(param->trainFname);
	cerr << "D=" << param->prob->D << endl;
	cerr << "K=" << param->prob->K << endl;
	cerr << "N=" << param->prob->N << endl;
	
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
	BDMMsolve* solver = new BDMMsolve(param);	
	solver->solve();
	//BCFWsolve* solver = new BCFWsolve(param);
	//Model* model = solver->solve();
	//writeModel(model, param);
	
	return 0;
}
