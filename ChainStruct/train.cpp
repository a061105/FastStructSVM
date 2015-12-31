#include "chain.h"

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

	return 0;
}
