
#ifndef SURFACEGRAPH_H
#define SURFACEGRAPH_H

#include <QtDataVisualization/Q3DSurface>
#include <QtDataVisualization/QSurfaceDataProxy>
#include <QtDataVisualization/QHeightMapSurfaceDataProxy>
#include <QtDataVisualization/QSurface3DSeries>
#include <QtWidgets/QSlider>
#include "GridCalculator.h"

using namespace QtDataVisualization;

enum class calc_grid_algo_type
{
	min_surface_algo,
	min_surface_with_tenes_algo,
	rbf_algo,
	kriging_algo
};


class SurfaceGraph : public QObject
{
	Q_OBJECT
	
public:
    explicit SurfaceGraph(Q3DSurface *surface);
    ~SurfaceGraph();

	void setBlackToYellowGradient();
    void setGreenToRedGradient();
    void setCalculType(int);
	void setRBFType(int);
	void setRBFParam(double);
	void setTension(double);
	void setKrigingType(int);
	void setKrigingAuto(bool);
	void calculate();

	template<int idx>
	void setKrigingParam(double val)
	{
		kriging_params[idx] = val;
	}

private:
    Q3DSurface *m_graph;
    QSurfaceDataProxy *proxy;
    QSurface3DSeries * series;

	QRectF r;
	double minZ{};
	double maxZ{};
	double tension = 0;
	unsigned char rbf_idx{};
	double rbf_param = 0.2;
	unsigned char  kriging_idx{};
	std::vector<double> kriging_params = {1,1,1};
	CalculParam  param{};
	bool auto_params=true;
	calc_grid_algo_type calcul_type{};
	Point3DList input_data;
	bool griding(CalculParam *calcul_grid);
	bool rbf_griding(CalculParam *calcul_grid);
	bool kriging_griding(CalculParam *calcul_grid);
	double step = 50.0;
	uint sampleCountX{};
	uint sampleCountZ{};
    void fillProxy();
	void open();



};

#endif // SURFACEGRAPH_H
