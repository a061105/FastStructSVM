#include "struct.h"
#include <cmath>
#include <iostream>
using namespace std;

double kernel(Param* param, SparseVect* x1, SparseVect* x2){

  //print(cerr,x1);
  //print(cerr,x2);
  double dist;
  if( param->kernel_type == 'G' )
    dist = dist_norm_sq(x1,x2);
  else if( param->kernel_type == 'L' )
    dist = dist_1norm(x1,x2);
  else{
    cerr << "unknown kernel type: " <<  param->kernel_type << endl;
    exit(0);
  }

  //cerr <<"dist_sq=" << exp(-param->gamma*dist_sq) << endl;
  return exp(-param->gamma*dist);
}

double kernel_inner_prod(Param* param, Model* model, SparseVect* xf, int yf, int ins_id, int factor_id){

  int node_id = model->offset[ins_id] + factor_id;
  SparseVect* alpha_is_yf = &(model->alpha[yf]);

  if( alpha_is_yf->size() == 0 )
    return 0.0;

  //cerr << "KV:" ;
  double prod = 0.0;
  for(SparseVect::iterator it=alpha_is_yf->begin(); it!=alpha_is_yf->end(); it++){
    double kv;
    kv = model->kernel_cache[node_id][it->first];
    if( kv < 0.0 ){
      kv = kernel(param, model->support_patterns->at(it->first), xf);
      model->kernel_cache[node_id][it->first] = kv;
    }
    //cerr << "(" << it->first << "," << kv << ")" << ", ";
    prod += it->second * kv;
  }
  //cerr << endl;

  return prod;
}
