#include "Solver.h"
#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
namespace
{
	constexpr double tol = 0.0000001;
}



bool LU_Solver(double *A, double *B, int n)
{
   
   auto *p_k = A;
   
   for (int k = 0; k < n; p_k += n, k++) 
   {
      
	   /*const int pivot =
		   tbb::parallel_reduce(tbb::blocked_range<int>(k + 1, n),qMa  ,
			  [&](const tbb::blocked_range<int>& r, MyMin current) -> std::pair<int, double>
			  {
				  for (int i = r.begin(); i < r.end(); ++i)
					  if (a[i] < current.value)
					  {
						  current.value = a[i]; current.idx = i;
					  }return current;
			  },
			  [](const MyMin a, const MyMin b) -> MyMin
	  {
		  return a.value < b.value ? a : b; }
	  );*/
			  
	  int pivot = k;
	  double *p_row = p_k + n;

	  for (int j = k + 1; j < n; j++, p_row += n)
	  {
	         
		 if (std::abs(*(p_row + k))>tol ) 
		 {
			pivot = j;
			break;
		 }
	  }
	
      if (pivot != k)
	  {
		  double *p_col		=	A+pivot*n;
		  std::swap(B[k],B[pivot]);

		  tbb::parallel_for(tbb::blocked_range<int>(0, n), [p_k, p_col](const tbb::blocked_range<int>&r)
			  {
				  for (int j = r.begin(); j < r.end(); ++j)
					  std::swap(*(p_k + j), *(p_col + j));
			  }
		  );

	  }


      if ( *(p_k + k) == 0.0 )
	  {
		  return {}; // матрица сингулярна
	  }
	  const double del=1/(*(p_k + k));

      for (int j = k+1; j < n; j++) 
	  {
			*(p_k + j)*= del;
      }

	  p_row = p_k + n;

      for (int i = k+1 ; i < n; p_row += n, i++)
	  {
		  const auto a= *(p_row + k);
		  for (int j = k+1; j < n; j++)
            *(p_row + j) -= a * *(p_k + j);
	  }

   }

   p_k = A;

   for (int k = 0 ; k < n; p_k += n, k++)
   {
	   for (int i = 0; i < k; i++) B[k] -= B[i] * *(p_k + i);
	   B[k] /= *(p_k + k);
   }

   p_k = A + n * (n - 1);

   for (int k = n - 1; k >= 0; k--, p_k -= n)
   {
	   for (int i = k + 1; i < n; i++) B[k] -= B[i] * *(p_k + i);
	   if (*(p_k + k) == 0.0) return {};
   }


   return true;
}


bool solveCrout::exec(std::vector<double> &S,std::vector<double> &b)
{
		return LU_Solver(&S[0], &b[0], b.size());
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
		const double delta=1/(*(p_i+i));
		
		for(int j = 0; j < n; ++j)
		{
			
            if(i!=j)
			{
				const double ratio = S[n*j+i]*delta;
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
		const double a = S[n*i+i];

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




