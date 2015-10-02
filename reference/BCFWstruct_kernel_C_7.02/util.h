#ifndef UTIL_H
#define UTIL_H

#include<fstream>
#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include <cmath>

using namespace std;

const int LINE_LEN = 10000000;
typedef vector<pair<int,double> > SparseVect;

double dist_norm_sq(SparseVect* v1, SparseVect* v2){
	
	SparseVect::iterator it = v1->begin();
	SparseVect::iterator it2 = v2->begin();
	
	double dist_sq = 0.0;
	while( it!=v1->end() && it2!=v2->end() ){
		
		if( it->first < it2->first ){
			dist_sq += it->second*it->second;
			it++;
		}
		else if( it2->first < it->first ){
			dist_sq += it2->second * it2->second;
			it2++;
		}
		else{
			dist_sq += (it->second - it2->second) * (it->second - it2->second);
			it++;
			it2++;
		}
	}

	if( it2!=v2->end() ){
		for(;it2!=v2->end();it2++)
			dist_sq += (it2->second)*(it2->second);
	}
	else if( it!=v1->end() ){
		for(;it!=v1->end();it++)
			dist_sq += (it->second)*(it->second);
	}

	return dist_sq;
}

double dist_1norm(SparseVect* v1, SparseVect* v2){
	
	SparseVect::iterator it = v1->begin();
	SparseVect::iterator it2 = v2->begin();
	
	double dist = 0.0;
	while( it!=v1->end() && it2!=v2->end() ){
		
		if( it->first < it2->first ){
			dist += fabs(it->second);
			it++;
		}
		else if( it2->first < it->first ){
			dist += fabs(it2->second);
			it2++;
		}
		else{
			dist += fabs(it->second - it2->second);
			it++;
			it2++;
		}
	}

	if( it2!=v2->end() ){
		for(;it2!=v2->end();it2++)
			dist += fabs(it2->second);
	}
	else if( it!=v1->end() ){
		for(;it!=v1->end();it++)
			dist += fabs(it->second);
	}

	return dist;
}

void sv_add(double s1, SparseVect* v1, double s2, SparseVect* v2, SparseVect* v3){
	
	SparseVect v_merge; v_merge.reserve(v1->size()+v2->size());
	
	SparseVect::iterator it = v1->begin();
	SparseVect::iterator it2 = v2->begin();
	
	while( it!=v1->end() && it2!=v2->end() ){
		
		if( it->first < it2->first ){
			v_merge.push_back( make_pair(it->first, s1*it->second) );
			it++;
		}
		else if( it2->first < it->first ){
			v_merge.push_back( make_pair(it2->first, s2*it2->second) );
			it2++;
		}
		else{
			v_merge.push_back( make_pair(it->first, s1*it->second + s2*it2->second) );
			it++;
			it2++;
		}
	}
	
	if( it2!=v2->end() ){
		for(;it2!=v2->end();it2++)
			v_merge.push_back( make_pair(it2->first, s2 * it2->second) );
	}
	else if( it!=v1->end() ){
		for(;it!=v1->end();it++)
			v_merge.push_back( make_pair(it->first, s1 * it->second)  ) ;
	}

	(*v3) = v_merge;
}

void sumReduce(SparseVect* list){
	
	sort(list->begin(), list->end());
	
	SparseVect* list2 = new SparseVect();

	int ind = -1;
	double val = 0.0;
	for(SparseVect::iterator it=list->begin();it!=list->end();it++){
		if( ind==it->first ){
			val += it->second;
		}else{
			if( ind != -1 )
				list2->push_back(make_pair(ind,val));
			
			ind = it->first;
			val = it->second;
		}
	}
	if( ind != -1 )
		list2->push_back(make_pair(ind,val));
	
	*list = *list2;
	
	delete list2;
}

void sumReduce(SparseVect* mat, int size){
	for(int i=0;i<size;i++){
		sumReduce(&(mat[i]));
	}
}

void mat_assign( SparseVect* s_arr1, SparseVect* s_arr2, int size ){
	
	for(int i=0;i<size;i++)
		s_arr1[i] = s_arr2[i];
}

void vassign( double* v1, double* v2, int size){
	
	for(int i=0;i<size;i++)
		v1[i] = v2[i];
}

void mat_times( SparseVect* s_arr, int size, double s){
	
	for(int i=0;i<size;i++)
		for(SparseVect::iterator it=s_arr[i].begin(); it!=s_arr[i].end(); it++){
			it->second *= s;
		}
}

void mat_times_restricted( SparseVect* s_arr, int size, double s, int lb, int ub){
	
	for(int i=0;i<size;i++)
		for(SparseVect::iterator it=s_arr[i].begin(); it!=s_arr[i].end(); it++){
			if( it->first >= lb && it->first < ub )
				it->second *= s;
		}
}

void mat_add( double s1, SparseVect* s_arr1, double s2, SparseVect* s_arr2, int size, SparseVect* s_arr3){
	
	for(int i=0;i<size;i++){
		sv_add( s1, &(s_arr1[i]), s2, &(s_arr2[i]), &(s_arr3[i]) );
	}
}

void vadd( double* v, double s,  SparseVect* a ){
	
	for(SparseVect::iterator it = a->begin(); it!=a->end(); it++){
		v[it->first] += s*it->second;
	}
}

void vtimes( double* v, int size, double s ){
	
	for(int i=0;i<size;i++)
		v[i] *= s;
}


vector<string> split(string str, string pattern){

	vector<string> str_split;
	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		str_split.push_back(str.substr(i,index-i));

		i = index+1;
	}
	
	if( str_split.back()=="" )
		str_split.pop_back();

	return str_split;
}

void print(ostream& out, SparseVect* mat, int size){
	
	for(int i=0;i<size;i++){
		if( mat[i].size() > 0 ){
			out << i << ": " ;
			for(int j=0;j<mat[i].size();j++){
				out << "(" << mat[i][j].first << "," << mat[i][j].second << "), ";
			}
			out << endl;
		}
	}
}

int nnz(SparseVect* mat, int size){
	
	int sum = 0;
	for(int i=0;i<size;i++){
		sum += mat[i].size();
	}
	return sum;
}

void print(ostream& out, SparseVect* vec){
	
	for(int j=0;j<vec->size();j++){
		out << "(" << vec->at(j).first << "," << vec->at(j).second << "), ";
	}
	out << endl;
}

void clear(SparseVect* mat, int size){
	
	for(int i=0;i<size;i++)
		mat[i].clear();
}

void readVect(char* fname, vector<int>& vect){
	
	vect.clear();
	
	ifstream fin(fname);
	int tmp;
	while( !fin.eof() ){
		fin >> tmp;
		if( fin.eof() )
			break;
		vect.push_back(tmp-1);
	}
	fin.close();
}

#endif
