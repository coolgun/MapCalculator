#pragma once
#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include <array>
#include "Solver.h"


using Point3DList=std::vector<std::tuple<double, double,double>> ;
using Point2DList = std::vector<std::pair<double,double>>;

struct CalculParam
{
	unsigned		width;
	unsigned		height;
	double			*node_values;
};


inline double tbb_dot(const size_t n, const double* x, const double* y)
{
	class  tbb_dot_summ
	{
		const double *x;
		const double *y;
		
	public:
		mutable double global_sum{};
		tbb_dot_summ(const double *px,const double *py):
			x(px),
			y(py),
			global_sum(0.0)

		{

		}
		void operator() ( const tbb::blocked_range<size_t>& r)const
		{				
			double sum=global_sum;
			for(size_t i= r.begin();i<r.end();++i)
			{
				sum+=x[i]*y[i];
			}
			global_sum=sum;
		}

		tbb_dot_summ(const tbb_dot_summ& x, tbb::split ) : 
			x(x.x), 
			y(x.y) ,
			global_sum(0.0)
		{

		
		}
 
	    void join( const tbb_dot_summ& x ) 
		{
			global_sum+=x.global_sum;
		}
	

};

	double sum=0.0;
	for(size_t i= 0;i<n;++i)
	{
		sum+=x[i]*y[i];
	}

	return sum;
	
}

// res <- mx
inline void mulMatrixVector(const size_t n, const double* m,const double* x ,double* res)
{
	
	class  mulMatrixVector_calcul
	{
			const double* m;
			const double* x;
			double		*res;
			const size_t n;

			
		public:
			mulMatrixVector_calcul(	size_t pN,	const double* pM,	const double* pX, double* pRes):
				n(pN),
				m(pM),
				x(pX),
				res(pRes)
			{
			}
			void operator() ( const tbb::blocked_range<size_t>& r)const
			{				
				
				double *pres	=	&res[r.begin()];
				auto  *prow	=	&m[r.begin()*n];
				for(size_t i= r.begin();i<r.end();++i,++pres)
				{
					  const double *px		=	x;
					  double sum =0.0;
					  for( size_t k=0; k<n; ++k,++prow,++px)
							sum += *prow**px;
					  *pres=sum;
				}

			}
	};

	const mulMatrixVector_calcul calc(n,m,x,res);
	tbb::parallel_for(tbb::blocked_range<size_t>(0, n),calc);
				
	
}




inline void SummingVector(const size_t n, const double* x,const double* y,double* res,double pA,double pB)
{
	
	class  tbb_over_calcul
	{
			const double* y;
			const double* x;
			double		*res;
			const double A;
			const double B;
			const size_t n;
			
		public:
			tbb_over_calcul(size_t pN,const double* pX,const double* pY,double* pRes,double pA,double pB):
				n(pN),
				x(pX),
				y(pY),
				res(pRes),
				A(pA),
				B(pB)

			{
			}
			void operator() ( const tbb::blocked_range<size_t>& r)const
			{				
				for(size_t i= r.begin();i<r.end();++i)
				{
					  res[i]=A*x[i]+B*y[i];
				}

			}
	};

	for(size_t i= 0;i<n;++i)
	{
			 res[i]=x[i]+pB*y[i];
	}
	
}

inline void SummingVector2(const size_t n, const double* x,const double* y,const double* z,double* res,double pA,double pB,double pC)
{
	
	class  tbb_over_calcul
	{
			const double* y;
			const double* x;
			const double* z;
			 double* res;
			const double		A;
			const double		B;
			const double		C;
			const size_t		 n;

			
		public:
			tbb_over_calcul(size_t pN,const double* pX,const double* pY,const double* pZ,double* pRes,double pA,double pB,double pC):
				n(pN),
				x(pX),
				y(pY),
				z(pZ),
				res(pRes),
				A(pA),
				B(pB),
				C(pC)
			{
			}
			void operator() ( const tbb::blocked_range<size_t>& r)const
			{				
				for(size_t i= r.begin();i<r.end();++i)
				{
					  res[i]=A*x[i]+B*y[i]+C*z[i];
				}

			}
	};


	for(size_t i= 0;i<n;++i)
	{
			res[i]=x[i]+pB*y[i]+pC*z[i];
	}

	
}


class solveBigStab
{
public:
	static void exec(std::vector<double> &A, std::vector<double> &b);
};




class GridInitializer
{
		public:
			virtual void Init(class CalculParam *calcul_grid,const  Point3DList &pList,struct Cher *)=0;


};

//struct CCalculGrid;

bool get_minmum_curvature		(CalculParam *calcul_grid,const  Point3DList &pList);
bool gaussian_rbf				(CalculParam *calcul_grid,const  Point3DList &pList,double param);
bool multiquadric_rbf			(CalculParam *calcul_grid,const  Point3DList &pList,double param);
bool revert_multiquadric_rbf	(CalculParam *calcul_grid,const  Point3DList &pList,double param);
bool polyharmonic_spline_rbf	(CalculParam *calcul_grid,const  Point3DList &pList,double power);
bool exp_kriging(CalculParam *calcul_grid, const  Point3DList &pList, bool is_auto, const std::vector<double> &C);
bool gauss_kriging(CalculParam *calcul_grid, const  Point3DList &pList, bool is_auto, const std::vector<double> &C);
bool sphere_kriging(CalculParam *calcul_grid, const  Point3DList &pList, bool is_auto, const std::vector<double> &C);
bool get_minmum_surface(CalculParam *calcul_grid, const  Point3DList &pList, double tension = 0.0, unsigned max_iteration = 2000, double max_miscalculation = 0.001, GridInitializer *initializer = {});



template<class Variogram> 
class FunctionFillingRBF
{	
			double *A;
			double	*B;
			const size_t		input_count;
			const size_t		size;
			const		Point3DList &list;
			const		Variogram	&var;
		public:
			FunctionFillingRBF(double *pA,double *pB,const  Point3DList &pList,const Variogram	&pVar):
				A(pA),
				B(pB),
				list(pList),
				input_count(pList.size()),
				size(pList.size()+3),
				var(pVar)
				
			{
			};
			
			void operator()( const tbb::blocked_range<size_t>& r ) const
	 		{
				for( size_t i=r.begin(); i!=r.end(); ++i )
				{
					const auto &[x_i, y_i, z_i]=list[i];
					for ( unsigned j=i; j<input_count; ++j )
					{
						const auto &[x_j, y_j, z_j] = list[j];
						double r2 = (x_j-x_i)*(x_j-x_i)+(y_j-y_i)*(y_j-y_i);
						A[(size)*i+j]=A[(size)*j+i]=var(r2);// r2*log((double)r2)*0.5;
					}
					B[i]=z_i;
					A[(size)*i+input_count+0]= A[(size)*input_count+i]	  =  1;
					A[(size)*i+input_count+1]= A[(size)*(input_count+1)+i]=  x_i;
					A[(size)*i+input_count+2]= A[(size)*(input_count+2)+i]=  y_i;
					
				}
			}
};


template<class Variogram> 
class FunctionCalculatorRBF
{	
			CalculParam *calcul_grid;
			const double *X;
			const int	input_count;
			const int	size;
			const		Point3DList &list;
			const double	a,b,c;
			const		Variogram	&var;

		public:
			FunctionCalculatorRBF(CalculParam *p_calcul_grid,double *pX,const  Point3DList &pList,const Variogram	&pVar):
				calcul_grid(p_calcul_grid),
				X(pX),
				list(pList),
				input_count(pList.size()),
				size(pList.size()+3),
				var(pVar),
				a(X[input_count]),
				b(X[input_count+1]),
				c(X[input_count+2])

			{
				
			};
			
			void operator()( const tbb::blocked_range2d<size_t>& r ) const
	 		{
				auto * p=calcul_grid;
				const size_t by=r.rows().begin();
				const size_t ey=r.rows().end();
				const size_t bx=r.cols().begin();
				const size_t ex=r.cols().end();
				size_t y_idx=by*p->width;

				for( size_t i=by; i!=ey; ++i,y_idx+=p->width  )
				{
					const double _y=i;
					size_t main_idx = y_idx+bx;
					
					for( size_t j=bx; j!=ex; ++j,main_idx++ ) 
					{
						
						const double _x			=	j;
						double h				=	a +b*_x+c*_y;
						const double *aa		=	X;
						 
						for(const auto &[x,y,z]: list )
						{
								const double x_i= x-_x;
								const double y_i= y-_y;
								const double r2 = x_i*x_i+y_i*y_i;
								h +=*aa*var(r2); //r2*log((double)r2)*0.5;
								++aa;
						
						}
						p->node_values[main_idx]=h;

					}
				
				}//for( size_t i=by; i!=ey; ++i )
		
		}//void operator()( const tbb::blocked_range2d<size_t>& r ) const
};




template<typename Variogram,typename Solver>
bool get_rbf(CalculParam *calcul_grid,const  Point3DList &pList,double param)
{
	
	const int size=pList.size()+3;
	if(size<3)  return false;
	std::vector<double> aA(size*size);
	std::vector<double> bB(size);
	auto *A=&aA[0];
	auto *B=&bB[0];

	Variogram var(calcul_grid,pList,param);

	tbb::parallel_for( 
					tbb::blocked_range<size_t>(0, pList.size())     
					,FunctionFillingRBF<Variogram>(A,B,pList,var)
					,tbb::auto_partitioner()
					);

	bool ret =Solver::exec(aA,bB);
	if(!ret) return false;
	tbb::parallel_for( 
				tbb::blocked_range2d<size_t>(0, calcul_grid->height, 0, calcul_grid->width)     
				,FunctionCalculatorRBF<Variogram>(calcul_grid,B,pList,var)
				,tbb::auto_partitioner()
				);
	return true;

}



template<class Variogram> 
class FunctionCalculatorKriging
{	
			CalculParam *calcul_grid;
			std::vector<double>		AyT;//vector
			std::vector<double>		AeT;//vector
			double				eAyT; 
			double				eAeT;
			const				Point3DList &list;
			const				Variogram	&var;
			const size_t		size;

		public:
			FunctionCalculatorKriging(CalculParam *p_calcul_grid,double *pA,const  Point3DList &pList,const Variogram	&pVar):
				calcul_grid(p_calcul_grid),
				list(pList),
				var(pVar),
				size(pList.size()),
				AyT(pList.size()),
				AeT(pList.size())

			{
					eAyT	=	0.0; 
					eAeT	=	0.0; 

					for(size_t i=0;i<size;++i)
					{		
							double		yT	=	0.0; 
							double		eT	=	0.0; 
							
							
							for(size_t j=0;j<size;++j)
							{	
								yT+=pA[size*i+j]*std::get<2>(pList[j]);	
								eT+=pA[size*i+j];
							}

							AyT[i]	=	yT;
							AeT[i]	=	eT;
							eAyT	+=yT;
							eAeT	+=eT;

					}

					eAyT=eAyT/eAeT;
					
				
			};
			
			void operator()( const tbb::blocked_range2d<size_t>& r ) const
	 		{
				auto *p=calcul_grid;
				const size_t by=r.rows().begin();
				const size_t ey=r.rows().end();
				const size_t bx=r.cols().begin();
				const size_t ex=r.cols().end();
				size_t y_idx=by*p->width;

				for( size_t i=by; i!=ey; ++i,y_idx+=p->width  )
				{
					const double _y=i;
					size_t main_idx = y_idx+bx;
					
					for( size_t j=bx; j!=ex; ++j,main_idx++ ) 
					{
						int	   k=0;
						double val	=	0.0;
						double val2	=	0.0;
						const double _x		=	j;
 
						for(const auto &[x,y,z]:list)
						{
								const double x_i= x	-_x;
								const double y_i= y -_y;
								const double r2 = x_i*x_i+y_i*y_i;
								double	v=	var(r2);
								val		+=v*AyT[k];
								val2	+=v*AeT[k];
								++k;
						}
						p->node_values[main_idx]=val-(val2-1)*eAyT;

					}
				
				}//for( size_t i=by; i!=ey; ++i )
		
		}//void operator()( const tbb::blocked_range2d<size_t>& r ) const
};



template<class Variogram> 
class FunctionFillingKriging
{	
			double		*A;
			const		size_t size;
			const		Point3DList &list;
			const		Variogram	&var;
		public:
			FunctionFillingKriging(double *pA,const  Point3DList &pList,const Variogram	&pVar):
				A(pA),
				list(pList),
				size(pList.size()),
				var(pVar)
				
			{
			};
			
			void operator()( const tbb::blocked_range<size_t>& r ) const
	 		{
				for( size_t i=r.begin(); i!=r.end(); ++i )
				{
					const auto&[x_i, y_i, z_i]=list[i];
					for (size_t j=i; j<size; ++j )
					{
						const auto&[x_j, y_j, z_j] = list[j];
						const double r2 = (x_j-x_i)*(x_j-x_i)+(y_j-y_i)*(y_j-y_i);
						A[(size)*i+j]=A[(size)*j+i]=r2<0.00000001?0.0:var(r2);
					}
				}
			}
};






template<typename Variogram,typename Ineverter>
bool get_kriging(CalculParam *calcul_grid,const  Point3DList &pList,bool is_auto,const std::vector<double> &C)
{
	int size=pList.size();
	if(size<3)  return false;
	std::vector<double> aA(size*size);


	Variogram var(calcul_grid,pList,is_auto,C);

	tbb::parallel_for( 
					tbb::blocked_range<size_t>(0, pList.size())     
					,FunctionFillingKriging<Variogram>(&aA[0],pList,var)
					,tbb::auto_partitioner()
					);

	const bool ret =Ineverter::invert(size, aA);
	if(!ret) return false;
	
	tbb::parallel_for( 
				tbb::blocked_range2d<size_t>(0, calcul_grid->height, 0, calcul_grid->width)     
				,FunctionCalculatorKriging<Variogram>(calcul_grid, &aA[0],pList,var)
				,tbb::auto_partitioner()
				);
	return true;

}

