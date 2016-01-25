#ifndef LP_SPARSE
#define LP_SPARSE

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include "LPutil.h"

using namespace std;

class LP_Param{
	
	public:

	char* data_dir;
	
	bool solve_from_dual;
	
	double tol;
	double tol_trans;
	double tol_sub;
	
	double eta;
	double nnz_tol;
	Int max_iter;

	LP_Param(){
		solve_from_dual = false;
		
		tol = 1e-2;
		//tol_trans = 10*tol;
		tol_trans = -1;
		tol_sub = 0.5*tol_trans;
		
		eta = 1.0;
		nnz_tol = 1e-4;
		max_iter = 1000;
	}
};	

LP_Param param;

/** Using Randomized Coordinate Descent to solve:
 *
 *  min c'x + \frac{eta_t}{2}\| (Ax-b+alpha_t/eta_t)_+ \|^2 + \frac{eta_t}{2}\|Aeq*x-t*beq+beta_t/eta_t\|^2
 *  s.t. x >= 0
 */
void rcd(Int n, Int nf, Int m, Int me, ConstrInv* At, double* b, double* c, double* x, double* w, double* h2_jj, double* hjj_ubound, double eta_t, Int& niter, Int inner_max_iter, Int& active_matrix_size, double& PGmax_old_last, Int phase){
	
	Int max_num_linesearch = 20;
	double sigma = 0.01; //for line search, find smallest t \in {0,1,2,3...} s.t.
			     // F(x+beta^t*d)-F(x) <= sigma * beta^t g_j*d
	
	//initialize active index
	Int* index = new Int[n+nf];
	Int active_size = n+nf;
	for(Int j=0;j<active_size;j++)
		index[j] = j;
	
	Int iter=0;
	double PG, PGmax_old = 1e300, PGmax_new;
	double d;
	while(iter < inner_max_iter){
		
		PGmax_new = -1e300;
		random_shuffle(index, index+active_size);
		
		for(Int s=0;s<active_size;s++){
			
			Int j = index[s];
			//cerr << "j=" << j << endl;
			
			//compute gradient, hessian of j-th coordinate
			double g = 0.0;
			double hjj = 0.0;
			for(ConstrInv::iterator it=At[j].begin(); it!=At[j].end(); it++){
				if( it->first < m && w[it->first] > 0.0 ){
					g += w[it->first] * it->second;
					hjj += it->second * it->second;
				}
				else if( it->first >= m ){
					g += w[it->first] * it->second;
				}
			}
			
			g *= eta_t;
			g += c[j];
			hjj *= eta_t;
			hjj += h2_jj[j];
			hjj = max( hjj, 1e-3);
			//hjj += 1.0;
			
			//compute PG and determine shrinking
			if( j>=n || x[j] > 0.0 ){
				
				PG = fabs(g);
			}else{
				PG = 0.0;
				//if( g > PGmax_old ){
				if( g > 0 ){
					active_size--;
					swap( index[s], index[active_size] );
					s--;
					continue;
				}else if( g < 0.0 ){
					PG = fabs(g);
				}
			}
			//cerr << "PG=" << PG << endl;
			
			if( PG > PGmax_new )
				PGmax_new = PG;
			if( PG < 1e-12 )
				continue;
			
			//compute d = Delta x
			if( j < n )
				d = max(x[j]-g/hjj, 0.0)-x[j];
			else
				d = -g/hjj;
			//cerr << "d=" << d << endl;
			
			//line search
			double d_old = 0.0, d_diff, rhs, cond, appxcond, fnew, fold;
			double delta = g*d;
			Int t;
			for(t=0; t<max_num_linesearch; t++){
				
				d_diff = d - d_old;
				
				cond = -sigma*delta;
				appxcond = hjj_ubound[j]*d*d/2.0 + g*d + cond;
				//cerr << "appxcond=" << appxcond << endl;
				if( appxcond <= 0.0 ){
					//update w, v
					for(ConstrInv::iterator it=At[j].begin();it!=At[j].end();it++)
						w[it->first] += d_diff * it->second;
					break;
				}
				
				if( t == 0 ){
					//compute fold, fnew (related to coordinate j)
					fold = c[j] * x[j];
					fnew = c[j] * (x[j] + d);
					double tmp_old=0.0, tmp_new=0.0;
					for(ConstrInv::iterator it=At[j].begin();it!=At[j].end();it++){
						//fold
						if( it->first >= m || w[it->first] > 0.0 )
							tmp_old += w[it->first]*w[it->first];
						//update w
						w[it->first] += d_diff * it->second;
						//fnew
						if( it->first >= m || w[it->first] > 0.0 )
							tmp_new += w[it->first]*w[it->first];
					}
					fold += eta_t*tmp_old/2.0;
					fnew += eta_t*tmp_new/2.0;
					
				}else{
					fnew = c[j] * (x[j]+d);
					double tmp = 0.0;
					for(ConstrInv::iterator it=At[j].begin();it!=At[j].end();it++){
						//update w
						w[it->first] += d_diff * it->second;
						if( it->first >= m || w[it->first] > 0.0 )
							tmp += w[it->first]*w[it->first];
					}
					fnew += eta_t*tmp/2.0;
				}
				
				cond += fnew - fold;
				if( cond <= 0 )
					break;
				else{
					d_old = d;
					d *= 0.5;
					delta *= 0.5;
				}
			}
			if( t == max_num_linesearch ){
				//cerr << "reach max_num_linesearch" << endl;
				return;
			}
			
			//update x_j
			x[j] += d;
		}
		iter++;
		if( iter % 10 == 0 ){
			//cerr << "." ;
		}
		
		PGmax_old = PGmax_new;
		if( PGmax_old <= 0.0 )
			PGmax_old = 1e300;
		
		if( PGmax_new <= param.tol_sub ){ //reach stopping criteria
			
			//cerr << "*" ;
			break;
		}
	}
	//passing some info
	niter = iter;
	active_matrix_size = 0;
	for(Int s=0;s<active_size;s++)
		active_matrix_size += At[index[s]].size();
	PGmax_old_last = PGmax_old;
	
	delete[] index;
}


/** Solve:
 *
 *  min  c'x
 *  s.t. Ax <= b
 *       Aeq x = beq
 *       x >= 0
 */
void LPsolve(Int n, Int nf, Int m, Int me, Constr* A, ConstrInv* At, double* b, double* c, double*& x, double*& w){
	
	double eta_t = param.eta;
	Int max_iter = param.max_iter;
	
	for(Int j=0;j<n+nf;j++)
		x[j] = 0.0;
	//w_t=Ax-b+w_{t-1}, v=Aeq*x-beq+ v_{t-1}
	for(Int i=0;i<m+me;i++)
		w[i] = -b[i]; //w=y/eta_t ==> w=-b+y/eta_t
	
	//initialize h2_ii (H=H1+H2) 
	double* h2_jj = new double[n+nf];
	for(Int j=0;j<n+nf;j++)
		h2_jj[j] = 0.0;
	for(Int i=m;i<m+me;i++){
		for(Constr::iterator it=A[i].begin(); it!=A[i].end(); it++){
			h2_jj[it->first] += it->second*it->second;
		}
	}
	for(Int j=0;j<n+nf;j++)
		h2_jj[j] *= eta_t;
	
	//initialize hjj's upper bound
	double* hjj_ubound = new double[n+nf];
	for(Int j=0;j<n+nf;j++)
		hjj_ubound[j] = 0.0;
	for(Int i=0;i<m;i++){
		for(Constr::iterator it=A[i].begin(); it!=A[i].end(); it++){
			hjj_ubound[it->first] += it->second*it->second;
		}
	}
	for(Int j=0;j<n+nf;j++){
		hjj_ubound[j] *= eta_t;
		hjj_ubound[j] += h2_jj[j];
	}
	//calculate matrix total # of nonzeros entries
	Int nnz_A = 0;
	for(Int i=0;i<n+nf;i++){
		nnz_A += At[i].size();
	}
	
	//main loop 
	double minus_time = 0.0;
	Int inner_max_iter=1;
	double PGmax_old = 1e300;
	Int niter;
	Int prInt_per_iter = 2;
	double pinf = 1e300, gap=1e300, dinf=1300, obj; //primal infeasibility
	Int nnz;//, nnz_last=1;
	Int active_nnz = nnz_A;
	Int phase = 1, inner_iter;
	double dinf_last=1e300, pinf_last=1e300, gap_last=1e300;
	for(Int t=0;t<max_iter;t++){
		
		if( phase == 1 ){
			inner_max_iter = (active_nnz!=0)?(nnz_A/active_nnz):(n+nf);
			//inner_max_iter = 1;
			rcd(n,nf,m,me, At,b,c,  x, w, h2_jj, hjj_ubound, eta_t, niter, inner_max_iter, active_nnz, PGmax_old, phase);
			
		}else if( phase == 2 ){
			
			rcd(n,nf,m,me, At,b,c,  x, w, h2_jj, hjj_ubound, eta_t, niter, inner_max_iter, active_nnz, PGmax_old, phase);
		}
		
		if( t % prInt_per_iter ==0 ){

			nnz = 0;
			for(Int j=0;j<n+nf;j++)
				if( x[j] > param.nnz_tol )
					nnz++;
			
			obj = 0.0;
			for(Int j=0;j<n+nf;j++)
				obj += c[j]*x[j];
			
			pinf = primal_inf(n,nf,m,me,x,A,b);
			dinf = dual_inf(n,nf,m,me,w,At,c,eta_t);
			gap = duality_gap(n,nf,m,me,x,w,c,b,eta_t);

			
			//cerr << setprecision(7) << "iter=" << t << ", #inner=" << niter << ", obj=" << obj ;
			//cerr << setprecision(2)  << ", p_inf=" << pinf << ", d_inf=" << dinf << ", gap=" << fabs(gap/obj) << ", nnz=" << nnz << "(" << ((double)active_nnz/nnz_A) << ")" ;
			//cerr << endl;
		}
		
		if( pinf<=param.tol && dinf<=param.tol ){
			break;
		}
		
		// w_t = Ax-b + w_{t-1} = w_t-1 + alpha_{t}/eta_t - alpha_{t-1}/eta_t  

		for(Int i=0;i<m;i++){ //inequality
			if( w[i] > 0 )
				w[i] -= b[i];
			else
				w[i] = -b[i];
		}
		for(Int i=m;i<m+me;i++){ //equality
			w[i] -= b[i];
		}
		
		for(Int j=0;j<n+nf;j++){ //both equality & inequality
			double tmp = x[j];
			for(ConstrInv::iterator it=At[j].begin(); it!=At[j].end(); it++)
				w[it->first] += tmp * it->second;
		}
		
		if( phase == 1 && pinf <= param.tol_trans ){
			
			phase = 2;
			prInt_per_iter = 1;
			//cerr << "phase = 2" << endl;
			inner_max_iter = 100;
		}
		
		if( phase == 2 ){
			
			if( (niter < 2) || (pinf < 0.5*dinf && dinf>dinf_last) ){
				
				if(niter < inner_max_iter) param.tol_sub *= 0.5;
				else			   inner_max_iter = min(inner_max_iter*2, (Int)10000);
				
			}
			
			if( dinf < 0.5*pinf && pinf > pinf_last ){
				
				//////////////////////////////////////correction
				for(Int i=0;i<m;i++){ //inequality
					if( w[i] > 0 )
						w[i] -= b[i];
					else
						w[i] = -b[i];
				}
				for(Int i=m;i<m+me;i++){ //equality
					w[i] -= b[i];
				}

				for(Int j=0;j<n+nf;j++){ //both equality & inequality
					double tmp = x[j];
					for(ConstrInv::iterator it=At[j].begin(); it!=At[j].end(); it++)
						w[it->first] += tmp * it->second;
				}
				//////////////////////////////////////////////
				eta_t *= 2;
				for(Int j=0;j<n+nf;j++){
					h2_jj[j] *= 2;
					hjj_ubound[j] *= 2;
				}
				for(Int i=0;i<m+me;i++)
					w[i] /= 2;
			}
			
		}

		dinf_last = dinf;
		pinf_last = pinf;
		gap_last = gap;
	}
	param.eta = eta_t;
}


#endif
