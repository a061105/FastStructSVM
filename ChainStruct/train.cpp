#include "chain.h"

int main(int argc, char** argv){
	
	if( argc < 1+1 ){
		cerr << "./train [data] (model)" << endl;
		exit(0);
	}

	char* fname = argv[1];
	char* modelFname;
	if( argc > 1+1 )
		modelFname = argv[2];
	else
		modelFname = "model";
	
	ChainProblem* prob = new ChainProblem();
	prob->readData(fname);
	
	return 0;
}
