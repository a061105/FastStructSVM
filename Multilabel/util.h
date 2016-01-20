#ifndef UTIL
#define UTIL

#include<cmath>
#include<vector>
#include<map>
#include<string>
#include<cstring>
#include<stdlib.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <omp.h>
#include <time.h>
#include <assert.h>
using namespace std;

typedef double Float;
typedef int Int;
typedef vector<pair<Int,Float> > SparseVec;
typedef vector<Int> Labels;
typedef pair<Float, Int>* ArrayHeap;
const Int LINE_LEN = 100000000;
const Int FNAME_LEN = 1000;

#define INFI 1e10
#define INIT_SIZE 16
#define PermutationHash HashClass
#define UPPER_UTIL_RATE 0.75
#define LOWER_UTIL_RATE 0.5

class ScoreComp{
	
	public:
	ScoreComp(Float* _score){
		score = _score;
	}
	bool operator()(const Int& ind1, const Int& ind2){
		return score[ind1] > score[ind2];
	}
	private:
	Float* score;
};

// Hash function [K] ->[m]

class HashFunc{
	
	public:
	Int* hashindices;
	HashFunc(){
	}
	HashFunc(Int _K){
		srand(time(NULL));
		K = _K;
		l = 10000;
		r = 100000;
		
		// pick random prime number in [l, r]
		p = rand() % (r - l) + l - 1;
		bool isprime;
		do {
			p++;
			isprime = true;
			for (Int i = 2; i * i <= p; i++){
				if (p % i == 0){
					isprime = false;
					break;
				}
			}
		} while (!isprime);
		a = rand() % p;
		b = rand() % p;
		c = rand() % p;
		hashindices = new Int[K];
		for (Int i = 0; i < K; i++){
			hashindices[i] = ((a*i*i + b*i + c) % p) % INIT_SIZE;
			if (i < INIT_SIZE) cerr << hashindices[i] % INIT_SIZE << " ";
		}
		cerr << endl;
	}
	~HashFunc(){
		delete [] hashindices;
	}
	void rehash(){
		p = rand() % (r - l) + l - 1;
                bool isprime;
                do {
                        p++;
                        isprime = true;
                        for (Int i = 2; i * i <= p; i++){
                                if (p % i == 0){
                                        isprime = false;
                                        break;
                                }
                        }
                } while (!isprime);
		a = rand() % p;
                b = rand() % p;
		for (Int i = 0; i < K; i++){
                        hashindices[i] = (a * i + b) % p;
                }
	}
	private:
	Int K, l, r;
	Int a,b,c,p;
};

class PermutationHash{
	public:
	PermutationHash(){};
	PermutationHash(Int _K){	
		srand(time(NULL));
		K = _K;
		hashindices = new Int[K];
		for (Int i = 0; i < K; i++){
			hashindices[i] = i;
		}
		random_shuffle(hashindices, hashindices+K);
	}
	~PermutationHash(){
		delete [] hashindices;
	}
	Int* hashindices;
	private:
	Int K;
};

vector<string> split(string str, string pattern){

	vector<string> str_split;
	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		if( index-i > 0 )
			str_split.push_back(str.substr(i,index-i));

		i = index+1;
	}
	
	if( str_split.back()=="" )
		str_split.pop_back();

	return str_split;
}

Float inner_prod(Float* w, SparseVec* sv){

	Float sum = 0.0;
	for(SparseVec::iterator it=sv->begin(); it!=sv->end(); it++)
		sum += w[it->first]*it->second;
	return sum;
}

Float prox_l1_nneg( Float v, Float lambda ){
	
	if( v < lambda )
		return 0.0;

	return v-lambda;
}

Float prox_l1( Float v, Float lambda ){
	
	if( fabs(v) > lambda ){
		if( v>0.0 )
			return v - lambda;
		else 
			return v + lambda;
	}
	
	return 0.0;
}

Float norm_sq( Float* v, Int size ){

	Float sum = 0.0;
	for(Int i=0;i<size;i++){
		if( v[i] != 0.0 )
			sum += v[i]*v[i];
	}
	return sum;
}

Int argmax( Float* arr, Int size ){
	
	Int kmax;
	Float max_val = -1e300;
	for(Int k=0;k<size;k++){
		if( arr[k] > max_val ){
			max_val = arr[k];
			kmax = k;
		}
	}
	return kmax;
}

//shift up, maintain reverse index
inline void siftUp(ArrayHeap heap, Int index, Int* rev_index){
	pair<Float, Int> cur = heap[index];
	while (index > 0){
		Int parent = (index-1) >> 1;
		if (cur > heap[parent]){
			heap[index] = heap[parent];
			rev_index[heap[parent].second] = index;
			index = parent;
		} else {
			break;
		}
	}
	rev_index[cur.second] = index;
	heap[index] = cur;
}

//shift down, maintain reverse index
inline void siftDown(ArrayHeap heap, Int index, Int* rev_index, Int size_heap){
	pair<Float, Int> cur = heap[index];
	Int lchild = index * 2 +1;
	Int rchild = lchild+1;
	while (lchild < size_heap){
		Int next_index = index;
		if (heap[lchild] > heap[index]){
			next_index = lchild;
		}
		if (rchild < size_heap && heap[rchild] > heap[next_index]){
			next_index = rchild;
		}
		if (index == next_index) 
			break;
		heap[index] = heap[next_index];
		rev_index[heap[index].second] = index;
		heap[next_index] = cur;
		index = next_index;
		lchild = index * 2 +1; rchild = lchild+1;
	}
	rev_index[cur.second] = index;
}

#endif
