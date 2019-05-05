#pragma once
#include <cmath>

template<int n> 
void solve_axb_cholesky(double l[n][n], double x[n], double b[n])
{
	  
	for (int i=0; i<n; i++) 
	{
		double sum = 0.0;
		for (int j=0; j<i; j++)
			sum += l[i][j] * x[j];
		x[i] = (b[i] - sum)/l[i][i];      
	}
 
	  for (int i=n-1; i>=0; i--) 
	  {
		double sum = 0.0;
		for (int j=i+1; j<n; j++)
		  sum += l[j][i] * x[j];
		x[i] = (x[i] - sum)/l[i][i];      
	  }
}
 

template<int n> 
int cholesky_decomp(double l[n][n], double a[n][n])
{
	  constexpr double TOL = 1e-30;
	  for (int i=0; i<n; i++) 
	  {
		for (int j=0; j<i; j++) 
		{
		  double sum = 0.0;
		  for (int k=0; k<j; k++)	
					sum += l[i][k] * l[j][k];
		  l[i][j] = (a[i][j] - sum)/l[j][j];
		}
 
		double sum = 0.0;
		for (int k=0; k<i; k++)
		  sum += l[i][k] * l[i][k];
    
		sum = a[i][i] - sum;
		if (sum<TOL) return 1; 
		l[i][i] = std::sqrt(sum);
	  }
	  return 0;
}



 
template<class Func>
double error_func(double *a, const std::vector<std::pair<double,double>> &pt, const Func &func)
{
	  double e=0.0;
	  for (const auto &iter:pt) 
	  {
		const auto res = func(a,std::get<0>(iter) - std::get<1>(iter));
		e += res*res;
	  }
	  return e;
}
 


template<int npar,class Func>
bool levmarq(double par[npar],const std::vector<std::pair<double,double>> &pt)
{
	int x	=	0;
	int i	=	0;
	int j	=	0;
	int it	=	0;
	int	nit	=	0;
	int	ill	=	0;
	double lambda,up,down,mult,weight,err,newerr,derr,target_derr;
	double h[npar][npar],ch[npar][npar];
	double g[npar],d[npar],delta[npar],newpar[npar];

	Func func(par,pt);
	
	nit		= 10000;
	lambda	= 0.0001;
	up		= 10.0;
	down		= 1/10.0;
    target_derr = 1e-12;
	weight = 1;
	derr = newerr = 0; 
	err = error_func<Func>(par,pt,func);
 
  
	  for (it=0; it<nit; it++) 
	  {
	 	for (i=0; i<npar; i++) 
		{
			d[i] = 0;
			for (j=0; j<=i; j++)	h[i][j] = 0;
		}

		for (const auto &[x,y] : pt)
		{
		  
		  func.grad(par,g, x);
	      
		  for (i=0; i<npar; i++) 
		  {
				d[i] += (y - func(par,x))*g[i];
				for (j=0; j<=i; j++)
				h[i][j] += g[i]*g[j];
		  }
		}
 
    
		mult = 1 + lambda;
		ill = true;

		while (ill && (it<nit)) 
		{
		  for (i=0; i<npar; i++)	
					h[i][i] = h[i][i]*mult;
	 
		  ill = cholesky_decomp<npar>(ch, h);
	 
		  if (!ill) 
		  {
				solve_axb_cholesky<npar>(ch, delta, d);
			
				for (i=0; i<npar; i++)
					newpar[i] = par[i] + delta[i];

				newerr = error_func<Func>(newpar,pt,func);
				derr = newerr - err;
				ill = (derr > 0);
		  } 

		  if (ill) 
		  {
				mult = (1 + lambda*up)/(1 + lambda);
				lambda *= up;
				it++;
		  }
		}
		for (i=0; i<npar; i++)     par[i] = newpar[i];
		err = newerr;
		lambda *= down;  
	 
		if ((!ill)&&(-derr<target_derr)) break;
	  }
	
	  return true;
}
 

 
