#include "Solver.h"
#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
namespace
{
	constexpr double tol = 0.0000001;
}

class  PivotFinder
{
		
	public:
		const double	*A;
		mutable double	max_val;
		int				k;
		mutable int		pivot;
		int				n;
		
		PivotFinder(int pN, const double *pA,int pK):
			n(pN),
			A(pA),
			k(pK),
			pivot(0),
			max_val(std::abs( pA[pN*pK+pK] ))

		{

		}
		void operator() ( const tbb::blocked_range<size_t>& r)const
		{				
			double	 m			=	max_val;
			int	 	 p			=	pivot;
			auto	 *pA		=	&A[n*(r.begin()+k+1)+k];
			
			for(size_t j= r.begin();j<r.end();++j,pA+=n)
			{
				const auto val=std::abs(*pA);
				if ( m < val ) 
				 {
					m = val;
					p = j+k+1;
				 }
			}
			max_val	=	m;
			pivot	=	p;
		}

		PivotFinder(const PivotFinder& x, tbb::split ) : 
			n(x.n),
			A(x.A),
			k(x.k),
			pivot(0),
			max_val(0.0)
		{

		
		}
 
	    void join( const PivotFinder& x ) 
		{
				if(max_val<x.max_val)
				{
					max_val		=	x.max_val;
					pivot		=	x.pivot;
				}

		}
	

};

	


bool Crout_LU_Decomposition_with_Pivoting(double *A, double *B, int n,bool *sign)
{
   int i, j, k; // p, row;
   double *p_k, *p_row, *p_col;
   double max;

   std::vector<double> tms(n);

   *sign=true;

   int pivot;

   PivotFinder piv_fnd(n,A,0);

   for (k = 0, p_k = A; k < n; p_k += n, k++) 
   {
      
 	  pivot=k;
			  
	  for (j = k + 1, p_row = p_k + n; j < n; j++, p_row += n)
	  {
	         
		 if (std::abs(*(p_row + k))>tol ) 
		 {
			pivot = j;
			break;
		 }
	  }
	
      if (pivot != k)
	  {
		  p_col		=	A+pivot*n;
		  
		  std::swap(B[k],B[pivot]);
		  *sign=!*sign;

		  for (j = 0; j < n; j++) 
		 {
            max = *(p_k + j);
            *(p_k + j) = *(p_col + j);
            *(p_col + j) = max;
         }
	  }

	  

      if ( *(p_k + k) == 0.0 )
	  {
		  return {}; // матрица сингулярна
	  }
	  double del=1/(*(p_k + k));

      for (j = k+1; j < n; j++) 
	  {
			*(p_k + j)*= del;
      }

      for (i = k+1, p_row = p_k + n; i < n; p_row += n, i++)
	  {
		  //*(p_k + j) /= *(p_k + k);
		  const auto a= *(p_row + k);
         
		  for (j = k+1; j < n; j++)
            *(p_row + j) -= a * *(p_k + j);
	  }

   }
   return true;
}

bool  Crout_LU_with_Pivoting_Solve(double *LU, double *B,  int n)

{
   int i, k;
   double *p_k;
   //double dum;

   std::vector<double> x(n);

   
   for (k = 0, p_k = LU; k < n; p_k += n, k++) 
   {
      x[k] = B[k];
      for (i = 0; i < k; i++) x[k] -= x[i] * *(p_k + i);
      x[k] /= *(p_k + k);
   }
   for (k = n-1, p_k = LU + n*(n-1); k >= 0; k--, p_k -= n)
   {
      for (i = k + 1; i < n; i++) x[k] -= x[i] * *(p_k + i);
	  if (*(p_k + k) == 0.0) return {};
   }

   std::copy_n(x.cbegin(), n, B);
  
   return true;
}



bool solveCrout::exec(std::vector<double> &S,std::vector<double> &b)
{
		bool sign=false;
		const int d = b.size();
		bool ret=Crout_LU_Decomposition_with_Pivoting(&S[0], &b[0],d,&sign);
		if(!ret) return false;
		ret=Crout_LU_with_Pivoting_Solve(&S[0],&b[0],d);
		if(!ret) return false;
		return true;
}

bool solveCrout::invert(int n,std::vector<double> &S)
{
	
	std::vector<double>  inv(n*n,0);
	
	for (int i = 0 ;i<n;++i) 	inv[n*i+i]=1;

	bool sign=false;
	 
	auto *p_i=&S[0];
	auto*i_i=&inv[0];

    for(int i = 0; i < n; ++i,p_i+=n,i_i+=n)
	{
		 int pivot=i;
		 auto *p_row=p_i;
		  
		  for (int j = i ;j < n; ++j,p_row += n)
		  {
			 if (std::abs(*(p_row + i))>tol ) 
			 {
				pivot = j;
				break;
			 }
		  }

		if (pivot != i)
		{
			sign=!sign;
			  
			for (int j = 0; j < n; ++j) 
			{
				std::swap(S[n*i+j],S[n*pivot+j]);
				std::swap(inv[n*i+j],inv[n*pivot+j]);
			}

		}
			
		if (!(*(p_i + i))) return {};
		double delta=1/(*(p_i+i));
		
		for(int j = 0; j < n; ++j)
		{
			
            if(i!=j)
			{
                double ratio = S[n*j+i]*delta;
                for(int k = 0; k <n; ++k)
				{
                    S[j*n+k]	-= ratio * S[n*i+k];
					inv[j*n+k]	-= ratio * inv[n*i+k];
				}
            }
        }
    }

    for(int	i = 0; i < n; i++)
	{
        double a = S[n*i+i];

        for(int	j = 0; j < n; ++j)
		{
            //S[n*i+j] /= a;
			if(sign)
				inv[n*i+j] /= -a;
			else
				inv[n*i+j] /= a;
        }
    }
	std::swap(S, inv);
	return true; 
		
}




