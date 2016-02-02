#include <stdlib.h>
#include <string>
#include <iostream>
using namespace std;

int main(int argc,char** argv){
	
	if( argc < 2 ){

		cerr << "Usage: ./pathToFile [path]" << endl;
		exit(0);
	}

	char* path = argv[1];
	string path_str(path);

	int index = path_str.find_last_of("/");
	
	if( index != path_str.size()-1 ){
		cout << path_str.substr(index+1) ;
	}

	return 0;
}
