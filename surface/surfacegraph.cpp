#include "surfacegraph.h"
#include <QtDataVisualization/QValue3DAxis>
#include <QtDataVisualization/Q3DTheme>
#include <QtSql/QSqlDatabase>
#include <QtSql/QSqlError>
#include <QtSql/QSqlQuery>
using namespace QtDataVisualization;

SurfaceGraph::SurfaceGraph(Q3DSurface *surface)
    : m_graph(surface)
{
	r.setCoords(std::numeric_limits<double>::max(), std::numeric_limits<double>::max(), 0.0, 0.0);
    m_graph->setAxisX(new QValue3DAxis);
    m_graph->setAxisY(new QValue3DAxis);
    m_graph->setAxisZ(new QValue3DAxis);
    proxy = new QSurfaceDataProxy();
    series = new QSurface3DSeries(proxy);
	open();
	fillProxy();

}

SurfaceGraph::~SurfaceGraph()
{
	delete []param.node_values;
    delete m_graph;
}

void SurfaceGraph::open()
{
	QSqlDatabase db;
	db = QSqlDatabase::addDatabase("QSQLITE");
	db.setDatabaseName("./read.db");
	db.open();
	QSqlQuery query(db);
	query.exec(QString("SELECT x,y,z  FROM xyz"));
	maxZ = {};
	while (query.next())
	{

		const auto x = query.value(0).toDouble();
		const auto y = query.value(1).toDouble();
		const auto z = query.value(2).toDouble();
		maxZ = qMax(maxZ, z);
		input_data.emplace_back(x, y, z);
		r.setCoords(qMin(r.left(), x), qMin(r.top(), y), qMax(r.right(), x), qMax(r.bottom(), y));
	}

	r.adjust(-r.width() / 10, -r.height() / 10, r.width() / 10, r.height() / 10);

	sampleCountX = std::round(r.width() / step);
	sampleCountZ = std::round(r.height() / step);
	r.setWidth(sampleCountX * step);
	r.setHeight(sampleCountZ  * step);

	for (auto &i : input_data)
	{

		std::get<0>(i) = (std::get<0>(i) - r.left()) / step;
		std::get<1>(i) = (std::get<1>(i) - r.top()) / step;

	}
	param=
	{
		sampleCountX,
		sampleCountZ,
		new double[sampleCountX*sampleCountZ]
	};
	maxZ *= 1.2;
	db.close();


	series->setDrawMode(QSurface3DSeries::DrawSurface);
	series->setFlatShadingEnabled(true);

	m_graph->axisX()->setLabelFormat("%.2f");
	m_graph->axisZ()->setLabelFormat("%.2f");
	m_graph->axisX()->setRange(r.left(), r.right());
	m_graph->axisY()->setRange(0.0f, maxZ);
	m_graph->axisZ()->setRange(r.top(), r.bottom());
	m_graph->axisX()->setLabelAutoRotation(30);
	m_graph->axisY()->setLabelAutoRotation(90);
	m_graph->axisZ()->setLabelAutoRotation(30);

	m_graph->addSeries(series);
	
}

void SurfaceGraph::fillProxy()
{

	griding(&param);

    QSurfaceDataArray *dataArray = new QSurfaceDataArray;
    dataArray->reserve(sampleCountZ);
	
	double z = r.top();


    for (int i = 0 ; i < sampleCountZ ; ++i,z+= step)
	{
        QSurfaceDataRow *newRow = new QSurfaceDataRow(sampleCountX);
    
		double x = r.left();
        for (int j = 0; j < sampleCountX; ++j, x+= step)
		{
			const auto val = qMax(0.0, param.node_values[i*sampleCountX + j]);
			//maxZ = qMax(maxZ, val);

            (*newRow)[j].setPosition(QVector3D(x, val,z));
        }
        *dataArray << newRow;
    }

    proxy->resetArray(dataArray);
}

void SurfaceGraph::calculate()
{
	fillProxy();
}

void SurfaceGraph::setRBFType(int val)
{
	rbf_idx = val;
}

void SurfaceGraph::setCalculType(int val)
{
	calcul_type = static_cast<calc_grid_algo_type>(val);
}


void SurfaceGraph::setRBFParam(double val)
{
	rbf_param = val;
}

void SurfaceGraph::setTension(double val)
{
	tension = val/100.0;
}

void SurfaceGraph::setKrigingType(int val)
{
	kriging_idx = val;
}

void SurfaceGraph::setKrigingAuto(bool val)
{
	auto_params = val;
}

//void setKrigingParam(double);


void SurfaceGraph::setBlackToYellowGradient()
{

    QLinearGradient gr;
    gr.setColorAt(0.0, Qt::black);
    gr.setColorAt(0.33, Qt::blue);
    gr.setColorAt(0.67, Qt::red);
    gr.setColorAt(1.0, Qt::yellow);

    m_graph->seriesList().at(0)->setBaseGradient(gr);
    m_graph->seriesList().at(0)->setColorStyle(Q3DTheme::ColorStyleRangeGradient);

}

void SurfaceGraph::setGreenToRedGradient()
{
    QLinearGradient gr;
    gr.setColorAt(0.0, Qt::transparent);
    gr.setColorAt(0.5, Qt::yellow);
    gr.setColorAt(0.8, Qt::red);
    gr.setColorAt(1.0, Qt::darkRed);

    m_graph->seriesList().at(0)->setBaseGradient(gr);
    m_graph->seriesList().at(0)->setColorStyle(Q3DTheme::ColorStyleRangeGradient);
}


///--------------------------------------------------------------------------------------------------------------------------------------

bool	SurfaceGraph::kriging_griding(CalculParam *calcul_grid)
{
	switch (kriging_idx)
	{
		case 0://"Экспоенента"
		{
			return exp_kriging(calcul_grid, input_data, auto_params, kriging_params);
		};
		case 1://Хаусс
		{
			return gauss_kriging(calcul_grid, input_data, auto_params, kriging_params);
		};
		case 2://Обр.квадрига 1/sqrt(1+r2/b
		{
			return sphere_kriging(calcul_grid, input_data, auto_params, kriging_params);
		};

	};

	return{};
}

///--------------------------------------------------------------------------------------------------------------------------------------

bool SurfaceGraph::rbf_griding(CalculParam *calcul_grid)
{

	switch (rbf_idx)
	{
		case 0://"Гауссиан exp(-r^2/b^2)"
		{
			return gaussian_rbf(calcul_grid, input_data, rbf_param);
		};
		case 1://Квадрига sqrt(1+r2/b)
		{
			return multiquadric_rbf(calcul_grid, input_data, rbf_param);
		};
		case 2://Обр.квадрига 1/sqrt(1+r2/b
		{
			return revert_multiquadric_rbf(calcul_grid, input_data, rbf_param);
		};
		case 3://Полигорманический сплайн
		{
			return polyharmonic_spline_rbf(calcul_grid, input_data, rbf_param);
		};

	};

	return{};
}

///--------------------------------------------------------------------------------------------------------------------------------------

bool SurfaceGraph::griding(CalculParam *calcul_grid)
{
	switch (calcul_type)
	{
		case calc_grid_algo_type::min_surface_algo:return get_minmum_curvature(calcul_grid, input_data);
		case calc_grid_algo_type::min_surface_with_tenes_algo:return get_minmum_surface(calcul_grid, input_data, tension);
		case calc_grid_algo_type::rbf_algo:return rbf_griding(calcul_grid);
		case calc_grid_algo_type::kriging_algo:return kriging_griding(calcul_grid);
	}
	return{};
}