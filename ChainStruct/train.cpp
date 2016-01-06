#include "chain.h"
#include "BDMMsolve.h"
#include "BCFWsolve.h"

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
	
	param->prob = new ChainProblem(param->trainFname);
	cerr << "D=" << param->prob->D << endl;
	cerr << "K=" << param->prob->K << endl;
	cerr << "N=" << param->prob->N << endl;
	cerr << "nSeq=" << param->prob->data.size() << endl;

	//BDMMsolve* solver = new BDMMsolve(param);	
	BCFWsolve* solver = new BCFWsolve(param);
	Model* model = solver->solve();
	//writeModel(model, param);
	
	return 0;
}
