#ifndef _CG_H
#define _CG_H

class Function
{
public:
	virtual void Hv(double *s, double *Hs) = 0 ;
	virtual int get_nr_variable(void) = 0 ;
	virtual ~Function(void){}
};

class CG
{
	public:
	CG(const Function *fun_obj, double eps = 0.1, int max_iter = 1000);
	~CG();
	
	void set_print_string(void (*i_print) (const char *buf));

	int cg(double *g, double *s, double *r);
	
	private:
	double norm_inf(int n, double *x);

	double eps;
	int max_iter;
	Function *fun_obj;
	void info(const char *fmt,...);
	void (*tron_print_string)(const char *buf);
};
#endif
