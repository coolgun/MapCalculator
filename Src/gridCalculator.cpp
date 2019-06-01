#include "GridCalculator.h"
#include "LevenbergMarquardt.h"




void ParallelMatrixMultiply(double *a, double *b, double *c,size_t size )
{
	tbb::parallel_for( tbb::blocked_range2d<size_t>(0, size, 16, 0, size, 32),     
    
		[a,b,c,size](const tbb::blocked_range2d<size_t>& r)
		{
			size_t y_idx = r.rows().begin()*size;
			for (size_t i = r.rows().begin(); i != r.rows().end(); ++i, y_idx += size)
			{
				size_t x_idx = y_idx + r.cols().begin();
				for (size_t j = r.cols().begin(); j != r.cols().end(); ++j, ++x_idx)
				{
					double sum = 0;
					size_t k_idx = 0;
					for (size_t k = 0; k < size; ++k, k_idx += size)
						sum += a[y_idx + k] * b[k_idx + j];

					c[x_idx] = sum;
				}
			}
		}
	
	);
}


class gaussian_variogram
{
	
	const double b;
public:
	gaussian_variogram(CalculParam*,const Point3DList&,double param)
		:b(param)
	{
		
	}
	

	double operator()(double r2) const
	{
		return std::exp(-r2/b/b);
	}
};


Point2DList calcul_semi_variogram(const Point3DList &list)
{
		
		Point2DList ret;

		for(int i=0; i<list.size(); ++i )
		{
			const auto &[x_i, y_i, z_i] =list[i];
			
			for (int j=i; j<list.size(); ++j )
			{
				const auto &[x_j, y_j, z_j] = list[j];
				const double r2 = (x_j-x_i)*(x_j-x_i)+(y_j-y_i)*(y_j-y_i);
				const double val = (z_i-z_j)*(z_i-z_j)/2.0;
				ret.push_back({ std::sqrt(r2),val });
			}
		}

		return ret;
}



class exp_kriging_variogram
{
		double a[3];
	public:

		class Func
		{
					
				public:

					Func(double *par,const std::vector<std::pair<double, double>> &pt)
					{
					
					}

					double operator()(double *a,double x) const
					{
						return  a[0]+ a[1]*(1-std::exp(-x/a[2]));
					}

					void grad(double *a,double *g, double x) const
					{
						g[0]	=	1;
						g[1]	=	(1-std::exp(-x/a[2]));
						g[2]	=	 -x*a[1]*std::exp(-x/a[2])/(a[2]*a[2]);
					}
		};
		
		exp_kriging_variogram(CalculParam *,const  Point3DList &pList,bool is_auto,const std::vector<double> &C)
		{
				
				if(is_auto)
				{
					a[0]=1;
					a[1]=1;
					a[2]=1;
					levmarq<3,Func>(a, calcul_semi_variogram(pList));
				}
				else
				{
					a[0]=C[0];
					a[1]=C[1];
					a[2]=C[2];
				}

		}

		double operator()(double r2) const
		{
			return  a[0]+a[1]*(1-std::exp(-std::sqrt(r2)/a[2]));
		}
};




bool exp_kriging(class CalculParam *calcul_grid,const  Point3DList &pList,bool is_auto,const std::vector<double> &C)
{
	return get_kriging<exp_kriging_variogram,solveCrout>(calcul_grid,pList,is_auto,C);
}


class gauss_kriging_variogram
{
		double a[3];
	public:

		class Func
		{
					
				public:

					Func(double *par,const std::vector<std::pair<double, double>> &pt)
					{
					
					}

					double operator()(double *a,double x) const
					{
						return  a[0]+a[1]*(1-std::exp(-x/a[2]));
					}

					void grad(double *a,double *g, double x) const
					{
						g[0]	=	1;
						g[1]	=	(1-std::exp(-x*x/a[2]));
						g[2]	=	 -x*x*a[1]*std::exp(-x*x/a[2])/(a[2]*a[2]);
					}
		};
		
		gauss_kriging_variogram(CalculParam *calcul_grid,const  Point3DList &pList,bool is_auto,const std::vector<double> &C)
		{
				
				if(is_auto)
				{
					a[0]=1;
					a[1]=1;
					a[2]=1;
					const auto  pt=calcul_semi_variogram(pList);
					levmarq<3,Func>(a,pt);
				}
				else
				{
					a[0]=C[0];
					a[1]=C[1];
					a[2]=C[2];
				}

		}

		double operator()(double r2) const
		{
			return  a[0]+a[1]*(1- std::exp(-r2/a[2]));
		}
};




bool gauss_kriging	(class CalculParam *calcul_grid,const  Point3DList &pList,bool is_auto,const std::vector<double> &C)
{
	return get_kriging<gauss_kriging_variogram,solveCrout>(calcul_grid,pList,is_auto,C);
}



class sphere_kriging_variogram
{
		double a[3];
	public:

		class Func
		{
					
				public:

					Func(double *par, const std::vector<std::pair<double, double>> &pt)
					{
					};
					
					double operator()(double *a,double x) const
					{
						const double tmp=x/a[2];
						return  a[0]+a[1]*(1.5*tmp-0.5*tmp*tmp*tmp);
					}

					void grad(double *a,double *g, double x) const
					{
						g[0]	=	1;
						const double tmp=x/a[2];
						g[1]	=	(1.5*tmp-0.5*tmp*tmp*tmp);
						g[2]	=	 a[1]*(-1.5*tmp/a[2]+1.5*tmp*tmp*tmp/a[2]);
					}
		};
		
		sphere_kriging_variogram(CalculParam *calcul_grid,const  Point3DList &pList,bool is_auto,const std::vector<double> &C)
		{
				
				
				if(is_auto)
				{
					a[0]=1;
					a[1]=1;
					a[2]=1;
					levmarq<3,Func>(a, calcul_semi_variogram(pList));
				}
				else
				{
					a[0]=C[0];
					a[1]=C[1];
					a[2]=C[2];
				}

		}

		double operator()(double r2) const
		{
			const double val=std::sqrt(r2);
			if(val>=a[2]) return a[0]+a[1];
			double tmp=val/a[2];
			return  a[0]+a[1]*(1.5*tmp-0.5*tmp*tmp*tmp);
		}
};




bool sphere_kriging	(class CalculParam *calcul_grid,const  Point3DList &pList,bool is_auto,const std::vector<double> &C)
{
	return get_kriging<sphere_kriging_variogram,solveCrout>(calcul_grid,pList,is_auto,C);
}




class multiquadric
{
	const double b;	
public:
	multiquadric(CalculParam *calcul_grid,const  Point3DList &pList,double param):
	  b(param)
	{
			

	}

	double operator()(double r2) const
	{
		return  std::sqrt(1+r2/b);
	}
};

class revert_multiquadric
{
	const double b;
public:
	revert_multiquadric(CalculParam *calcul_grid,const  Point3DList &pList,double param):
	  b(param)
	{
			

	}

	double operator()(double r2) const
	{
		return  1/std::sqrt(1+r2/b);
	}
};



class plate_spline 
{
public:
	plate_spline (CalculParam *calcul_grid,const  Point3DList &pList,double )
	{
	
	}

	double operator()(double r2) const
	{
		return r2?r2*std::log(r2)*0.5:0;
	}
};





class polyharmonic_spline
{
	const int n;
public:
	polyharmonic_spline (CalculParam *calcul_grid,const  Point3DList &pList,double power):
	  n(std::round( power))
	{
	
	}

	double operator()(double r2) const
	{
		if (!r2) return {};
		const double r=std::sqrt(r2);
		return n%2?std::pow(r,n):std::pow(r,n)*std::log(r);
		
	}
};





bool get_minmum_curvature(CalculParam *calcul_grid,const  Point3DList &pList)
{
	return get_rbf<plate_spline ,solveCrout>(calcul_grid,pList,0);
	
}

bool gaussian_rbf(CalculParam *calcul_grid,const  Point3DList &pList,double b)
{

	return get_rbf<gaussian_variogram ,solveCrout>(calcul_grid,pList,b);
	
}

bool multiquadric_rbf(CalculParam *calcul_grid,const  Point3DList &pList,double param)
{

	return get_rbf<multiquadric ,solveCrout>(calcul_grid,pList,param);
	
}

bool revert_multiquadric_rbf(CalculParam *calcul_grid,const  Point3DList &pList,double param)
{

	return get_rbf<revert_multiquadric ,solveCrout>(calcul_grid,pList,param);
	
}


bool polyharmonic_spline_rbf(class CalculParam *calcul_grid,const  Point3DList &pList,double power)
{

	return get_rbf<polyharmonic_spline ,solveCrout>(calcul_grid,pList,power);
	
}




double Gaussina(
					double x,
					double y,
					double xsr,
					double ysr,
					double zsum,
					double Dx,
					double Dy,
					double corr,
					double corrective_factor
				  )
{
	return corrective_factor*std::exp
					(	-0.5f/(1-corr*corr)*
					(
					   (x-xsr)*(x-xsr)/Dx+
					   (y-ysr)*(y-ysr)/Dy-
						2.0f*corr*(y-ysr)*(x-xsr)/ std::sqrt(Dx*Dy)
				 	)
					)
					/(2*3.1415f*std::sqrt(Dx*Dy*(1-corr*corr)));
	

	
}


void get_gaussian(
					double &xsr,
					double &ysr,
					double &zsum,
					double &Dx,
					double &Dy,
					double &corr,
					double &corrective_factor,
					const  Point3DList &pList
				  )
{
   xsr=0;
   ysr=0;
   zsum=0;
   Dx=0;
   Dy=0;
   double Cov=0;
   for(auto &[x, y, z] :pList)
   {
	   double tmp_x=x*z;
	   xsr+=tmp_x;
	   Dx+=tmp_x*x;
	   double tmp_y=y*z;
	   ysr+=tmp_y;
	   Dy+=tmp_y*y;
	   Cov+=tmp_x*y;
	   zsum+=z;
   }
    xsr/=zsum;
	ysr/=zsum;
	Dx=(Dx)/zsum-xsr*xsr;
	Dy=(Dy)/zsum-ysr*ysr;
	Cov=Cov/zsum-xsr*ysr;
	corr=Cov/(std::sqrt(Dx*Dy));//коэфф кореляции;
	double tmp1=0;
	double tmp2=0;

	for (const auto &[x, y, z] : pList)
	{
		double dx=x-xsr;
		double dy=y-xsr;

		double tmp_f= std::exp
				(	-0.5/(1-corr*corr)*
					(
					   dx*dx/Dx+dy*dy/Dy-
						2*corr*(dy-ysr)*(dx-xsr)/ std::sqrt(Dx*Dy)
				 	)
					)
			/(2*3.1415f*std::sqrt(Dx*Dy*(1-corr*corr)));
		tmp1+=tmp_f*tmp_f;
		tmp2+=z;
	}
	
	corrective_factor=tmp2/tmp1;

}


struct Cher
{
	bool flag{};
	double r{};
	double p{};
	double Ap{};
	double node_value{};

};


struct AdvData
{
	size_t	two_node_x;
	double tmp_t;
	double tmp_t_2;
	double tt;
};

class GaussionInitializer:public  GridInitializer
{

	public:
			virtual void Init(CalculParam *calcul_grid,const  Point3DList &pList,Cher *cher)
			{
					
					int		node_x_count	=	calcul_grid->width;
					int		node_y_count	=	calcul_grid->height;
					const unsigned size=node_x_count*node_y_count;

					double xsr,ysr,zsum,Dx,Dy,corr,corrective_factor;
					get_gaussian(xsr,ysr,zsum,Dx,Dy,corr,corrective_factor,pList);
					for(int x=0;x<node_x_count+4;++x)
					{
						for(int y=0;y<node_y_count+4;++y)
						{
							int idx=y*(node_x_count+4)+x;
							cher[idx].node_value=Gaussina(x-2,y-2,xsr,ysr,zsum,Dx,Dy,corr,corrective_factor  );
						}
					}
	

					for(int x=0;x<node_x_count+4;++x)
					{
						int idx=x;	
						cher[idx].flag=true;
						idx=(node_x_count+4)+x;	
						cher[idx].flag=true;
						idx=(node_y_count-1+4)*(node_x_count+4)+x;
						cher[idx].flag=true;
						idx=(node_y_count-2+4)*(node_x_count+4)+x;
						cher[idx].flag=true;

					}

					for(int y=0;y<node_y_count+4;++y)
					{
						int idx=(node_x_count+4)*y;	
						cher[idx].flag=true;
						idx=(node_x_count+4)*y+1;	
						cher[idx].flag=true;
						idx=y*(node_x_count+4)+(node_x_count+4)-1;
						cher[idx].flag=true;
						idx=y*(node_x_count+4)+node_x_count-2+4;
						cher[idx].flag=true;
					}
	

			}
	
};

class VoronoiInitializer:public  GridInitializer
{

	
	public:
			VoronoiInitializer() = default;
			

			virtual void Init(CalculParam *calcul_grid,const  Point3DList &pList,Cher *cher)
			{
					const	int		node_x_count	=	calcul_grid->width+4;
					const	int		node_y_count	=	calcul_grid->height+4;
					const unsigned size=node_x_count*node_y_count;
					std::vector<int>		voron(node_x_count*node_y_count,-1);
					tbb::concurrent_vector<int>		pWave;
					tbb::concurrent_vector<int>		pTmpWave;

					tbb::parallel_for(tbb::blocked_range<Point3DList::const_iterator>(pList.cbegin(), pList.cend()),
						[cher,node_x_count, node_y_count, &voron,&pWave](const tbb::blocked_range<Point3DList::const_iterator> &r)
						{
							std::vector<int> tmp;
							for (auto i = r.begin(); i != r.end(); ++i)
							{
								const auto &[x, y, z] = *i;
								int x_idx = std::round(x) + 2;
								if (x_idx < 0 || x_idx >= node_x_count) continue;
								const int y_idx = std::round(y) + 2;
								if (y_idx < 0 || y_idx >= node_y_count) continue;
								const int idx = y_idx * (node_x_count)+x_idx;
								if (idx < 0) continue;
								voron[idx] = idx;
								cher[idx].node_value = z;
								tmp.push_back(idx);
							}
							std::copy(tmp.cbegin(), tmp.cend(), pWave.grow_by(tmp.size()));
							
						}
					);
					
					double xsr, ysr, zsum, Dx, Dy, corr, corrective_factor;
					get_gaussian(xsr, ysr, zsum, Dx, Dy, corr, corrective_factor, pList);

					
					for(int x=0;x<node_x_count;++x)
					{
						int idx=x;	
						cher[idx].flag=true;
						//cher[idx].node_values = 0;
						//cher[idx].node_values = Gaussina(x - 2, y - 2, xsr, ysr, zsum, Dx, Dy, corr, corrective_factor);
						
						idx=(node_x_count)+x;	
						cher[idx].flag=true;
						//cher[idx].node_values = 0;
						
						idx=(node_y_count-1)*(node_x_count)+x;
						cher[idx].flag=true;
						//cher[idx].node_values = 0;
						
						idx=(node_y_count-2)*(node_x_count)+x;
						cher[idx].flag=true;
						//cher[idx].node_values = 0;


					}

					for(int y=0;y<node_y_count;++y)
					{
						int idx=(node_x_count)*y;	
						cher[idx].flag=true;
						//cher[idx].node_values = 0;
						
						idx=(node_x_count)*y+1;	
						cher[idx].flag=true;
						//cher[idx].node_values = 0;
						
						
						idx=y*(node_x_count)+(node_x_count)-1;
						cher[idx].flag=true;
						//cher[idx].node_values = 0;

						idx=y*(node_x_count)+node_x_count-2;
						cher[idx].flag = true;
						//cher[idx].node_values = 0;
					}
	
					while (!pWave.empty())
					{
						pTmpWave.clear();

						tbb::parallel_for(tbb::blocked_range<tbb::concurrent_vector<int>::const_iterator>(pWave.cbegin(), pWave.cend()),
							[cher, node_x_count, node_y_count, &voron, &pTmpWave](tbb::blocked_range<tbb::concurrent_vector<int>::const_iterator> &r)
						{
							constexpr std::array<std::pair<int, int>, 8> plus_minus =
							{
								std::make_pair(-1,-1),
								std::make_pair(-1,0),
								std::make_pair(-1,+1),
								std::make_pair(0,-1),
								std::make_pair(0,+1),
								std::make_pair(1,-1),
								std::make_pair(1,0),
								std::make_pair(1,+1)
							};

							for (auto i = r.begin(); i != r.end(); ++i)
							{
								const auto idx = *i;
								const int v = voron[idx];
								const int y = idx / node_x_count;
								const int x = idx % node_x_count;

								for (const auto &[xd, yd] : plus_minus)
								{
									const int _y = y + yd;
									const int _x = x + xd;
									const int	tmp_idx = _y * node_x_count + _x;
									if (_x < 0 || _y < 0 || _x >= node_x_count || _y >= node_y_count) continue;
									int	&_v = voron[tmp_idx];

									if (_v < 0)
									{
										_v = v;
										cher[tmp_idx].node_value = cher[v].node_value;

										int y2 = v / node_x_count;
										int x2 = v % node_x_count;

										if (x2 == 1 || y2 == 1 || x2 == node_x_count - 2 || y2 == node_y_count - 2)
										{
											cher[tmp_idx].flag = true;

										}

										pTmpWave.push_back(tmp_idx);
										continue;
									}


								}

							}
						}
						);

						std::swap(pWave, pTmpWave);

					}

					
					
					tbb::parallel_for(tbb::blocked_range<Point3DList::const_iterator>(pList.cbegin(), pList.cend()),
						[cher, node_x_count, node_y_count](const tbb::blocked_range<Point3DList::const_iterator> &r)
						{
							for (auto i = r.begin(); i != r.end(); ++i)
							{
								const auto &[x, y, z] = *i;
								const int x_idx = std::round(x) + 2;
								if (x_idx < 0 || x_idx >= node_x_count) continue;
								const int y_idx = std::round(y) + 2;
								if (y_idx < 0 || y_idx >= node_y_count) continue;
								const int idx = y_idx * (node_x_count)+x_idx;
								if (idx < 0) continue;
								cher[idx].node_value = z;
								cher[idx].flag = true;
							}
						}
					);
			
			}
	
};




bool get_minmum_surface(CalculParam *calcul_grid,const  Point3DList &pList,double tension,unsigned max_iteration,double max_miscalculation,GridInitializer *initializer /*=0 */)
{
	
	const int		node_x_count	=	calcul_grid->width+4;
	const int		node_y_count	=	calcul_grid->height+4;
	double	*const node_values	=	calcul_grid->node_values;
	const unsigned size		=	(node_x_count)*(node_y_count);

	std::vector<Cher>  cCher(size);
	auto * const cher=&cCher[0];

	if(initializer)
	{
		initializer->Init(calcul_grid,pList,cher);
	}
	else
	{
	//	GaussionInitializer in;
		VoronoiInitializer in;
		in.Init(calcul_grid,pList,cher);
	}

	tbb::parallel_for(tbb::blocked_range<Point3DList::const_iterator>(pList.cbegin(), pList.cend()),
		[cher, node_x_count, node_y_count](const tbb::blocked_range<Point3DList::const_iterator> &r)
		{
			for (auto i = r.begin(); i != r.end(); ++i)
			{
				const auto &[x, y, z] = *i;
				const int x_idx = std::round(x) + 2;
				if (x_idx < 0 || x_idx >= node_x_count) continue;
				const int y_idx = std::round(y) + 2;
				if (y_idx < 0 || y_idx >= node_y_count) continue;
				const int idx = y_idx * (node_x_count)+x_idx;
				if (idx < 0) continue;
				cher[idx].node_value = z;
				cher[idx].flag = true;

			}
		}
	);


	double rr = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, size), 0.0,
		[cher, node_x_count, tension](const tbb::blocked_range<size_t>& r, double rr)->double
	{

		for (auto k = r.begin(); k != r.end(); ++k)
		{
			if (cher[k].flag) continue;
			cher[k].r = -((1 - tension)*
				(
					12 * cher[k].node_value -
					4 * (
						cher[k - 1].node_value +
						cher[k + 1].node_value +
						cher[k - node_x_count].node_value +
						cher[k + node_x_count].node_value
						) +
						(
							cher[k - 2].node_value +
							cher[k + 2].node_value +
							cher[k - 2 * node_x_count].node_value +
							cher[k + 2 * node_x_count].node_value
							)
					) +
				tension * (4 * cher[k].node_value - cher[k - 1].node_value - cher[k + 1].node_value - cher[k - node_x_count].node_value - cher[k + node_x_count].node_value)
				);

			cher[k].p = cher[k].r;
			rr += cher[k].r*cher[k].r;
		}
		return rr;
	},
	std::plus<double>()
	);

	
	const double max_miscalculation_quad=max_miscalculation*max_miscalculation;
	const AdvData adv_data={ 2 * node_x_count,12 - 8 * tension,4 - 3 * tension,1 - tension };
	
	for(unsigned  step = 0;step<max_iteration;++step)
	{
		
		const auto App = tbb::parallel_reduce(tbb::blocked_range<size_t>(0,size-4*node_x_count),0.0, 
			[cher, node_x_count, adv_data](const tbb::blocked_range<size_t>& r, double init)->double
			{
				auto  *point_cher = cher + adv_data.two_node_x + r.begin();
				for (size_t i = r.begin(); i < r.end(); ++i, ++point_cher)
				{
					if (point_cher->flag) 	continue;
					double tmp_sum = (point_cher - 1)->p + (point_cher + 1)->p + (point_cher - node_x_count)->p + (point_cher + node_x_count)->p;
					point_cher->Ap = adv_data.tmp_t*point_cher->p + adv_data.tt*(
						(point_cher - 2)->p +
						(point_cher + 2)->p +
						(point_cher - adv_data.two_node_x)->p +
						(point_cher + adv_data.two_node_x)->p
						) - adv_data.tmp_t_2*tmp_sum;
					init += point_cher->Ap*point_cher->p;
				}
				return init;
			},
			std::plus<double>()
		);
		
		const auto alpha=rr/App;
		const auto new_rr = tbb::parallel_reduce(tbb::blocked_range<size_t>(0, size - 4 * node_x_count),0.0,
			[cher, node_x_count, alpha](const tbb::blocked_range<size_t>& r,double init)->double
			{
				
				auto  *point_cher = cher + 2 * node_x_count + r.begin();

				for (auto i = r.begin(); i < r.end(); ++i, ++point_cher)
				{
					if (point_cher->flag) 	continue;

					point_cher->node_value += alpha * point_cher->p;
					point_cher->r -= alpha * point_cher->Ap;
					init += point_cher->r*point_cher->r;
				}
				return init;
			},
			std::plus<double>()
		);


		if(new_rr<(max_miscalculation_quad)) 	
			break;
		const auto beta=new_rr/rr;
		rr=new_rr;

		tbb::parallel_for
		(
			tbb::blocked_range<size_t>(0,size-4*node_x_count),
			[cher, node_x_count, beta](const tbb::blocked_range<size_t>& r)
			{
				auto  *point_cher = cher + 2 * node_x_count + r.begin();

				for (auto i = r.begin(); i < r.end(); ++i, ++point_cher)
				{
					if (point_cher->flag) 		continue;
					(point_cher->p *= beta) += point_cher->r;
				}
			}
		);
	}

	tbb::parallel_for(tbb::blocked_range2d<size_t>(0, calcul_grid->height, 16, 0, calcul_grid->width, 32),
		[node_values, cher, calcul_grid, node_x_count](const tbb::blocked_range2d<size_t>& r)
		{

			for (auto y = r.rows().begin(); y != r.rows().end(); ++y)
			{
				for (auto x = r.cols().begin(); x != r.cols().end(); ++x)
				{
					node_values[y*calcul_grid->width + x] = cher[(y + 2)*node_x_count + x + 2].node_value;
				}
			}

		}
	);
	
	return true;

}



void tbb_over(const size_t n, double* u,const double* v, const double *alpha)
{
	tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
		[u, v, alpha](const tbb::blocked_range<size_t>& r)
			{
				for (size_t i = r.begin(); i < r.end(); ++i)
				{

					u[i] = u[i] + alpha[i] * (v[i] - u[i]);

				}
			}
		);
	
}




void solveBigStab::exec(std::vector<double> &A, std::vector<double> &b)
{
	const int n = b.size();
	double max_v=0;
	for(int i=0;i< n;++i)
	{
		for(int j=0;j< n;++j)
		{
				max_v=std::max(max_v,std::abs(A[i*n +j]));
		}
	}

	for(int i=0;i< n;++i)
	{
		for(int j=0;j<n;++j)
		{
			A[i*n +j]/=max_v;
		}
		b[i]/=max_v;
	};

	
	std::vector<double>		 xx=b;
	std::vector<double>		 y(n,0.0);
	std::vector<double>      r(n,0.0);
	std::vector<double>      s(n,0.0);
	std::vector<double>      AMp(n,0.0);
    
	std::vector<double>      AMs(n,0.0);

    // y <- Ax
	mulMatrixVector(n,&A[0],&xx[0] ,&y[0]);
    
    // r <- b - A*x
	SummingVector(n,&b[0],&y[0],&r[0],1,-1);

	auto p = r;
	auto r_star = r;
    double r_r_star_old = tbb_dot(n,&r_star[0], &r[0]);

	
	int iter=0; 
	double  tmp=0.0;

    while ( (tmp=tbb_dot(n,&r[0], &r[0]))>10e-9)
    {
      
		mulMatrixVector(n, &A[0],&p[0] ,&AMp[0]);

        // alpha = (r_j, r_star) / (A*M*p, r_star)
		const double alpha = r_r_star_old / tbb_dot(n,&r_star[0], &AMp[0]);
        
        // s_j = r_j - alpha * AMp
        
		SummingVector(n,&r[0],&AMp[0],&s[0],1,-alpha);

		tmp=tbb_dot(n,&s[0], &s[0]);

		if (tmp<10e-9)
		{
		  // x += alpha*M*p_j
		  SummingVector(n,&xx[0],&p[0], &xx[0],1,alpha);
		  break;
		}

        // Ms = M*s_j
		
        
        // AMs = A*Ms
		mulMatrixVector(n, &A[0],&s[0] ,&AMs[0]);

        // omega = (AMs, s) / (AMs, AMs)
		const double omega = tbb_dot(n,&AMs[0], &s[0]) / tbb_dot(n,&AMs[0], &AMs[0]);
        
        // x_{j+1} = x_j + alpha*M*p_j + omega*M*s_j
		SummingVector2(n, &xx[0],&p[0], &s[0], &xx[0],1,alpha, omega);

        // r_{j+1} = s_j - omega*A*M*s
		SummingVector(n,&s[0],&AMs[0],&r[0],1,-omega);

        // beta_j = (r_{j+1}, r_star) / (r_j, r_star) * (alpha/omega)
		const double	 r_r_star_new = tbb_dot(n,&r_star[0], &r[0]);
		const double	 beta = (r_r_star_new / r_r_star_old) * (alpha / omega);
        r_r_star_old = r_r_star_new;

        // p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
		SummingVector2(n,&r[0],&p[0], &AMp[0], &p[0],1,beta, -beta*omega);
    }

	std::swap(b, xx);

}


