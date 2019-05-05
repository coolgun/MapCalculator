
#include "surfacegraph.h"
#include <QtWidgets/QApplication>
#include <QtWidgets/QWidget>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QRadioButton>
#include <QtWidgets/QSlider>
#include <QtWidgets/QGroupBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QSpinBox>
#include <QtWidgets/QLabel>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QMessageBox>
#include <QtGui/QPainter>
#include <QtGui/QScreen>

int main(int argc, char **argv)
{
    QApplication app(argc, argv);
    auto *graph = new Q3DSurface();
    auto *container = QWidget::createWindowContainer(graph);

    if (!graph->hasContext()) 
	{
        QMessageBox msgBox;
        msgBox.setText("Couldn't initialize the OpenGL context.");
        msgBox.exec();
        return -1;
    }

    const auto screenSize = graph->screen()->size();
    container->setMinimumSize(QSize(screenSize.width() / 2, screenSize.height() / 1.6));
    container->setMaximumSize(screenSize);
    container->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    container->setFocusPolicy(Qt::StrongFocus);

    
	auto *widget = new QWidget;
	auto *hLayout = new QHBoxLayout(widget);
	auto *vLayout = new QVBoxLayout();
    hLayout->addWidget(container, 1);
    hLayout->addLayout(vLayout);
    vLayout->setAlignment(Qt::AlignTop);
    widget->setWindowTitle(QStringLiteral("Grid calculator"));

	QVector<QPair<int, QWidget*>> controls;


    auto *calc_type = new QComboBox();
	calc_type->addItems
	(
		{
			QStringLiteral("spline"),
			QStringLiteral("difference scheme"),
			QStringLiteral("radial basis function"),
			QStringLiteral("kriging")
		}
	);

	auto *teneson = new QDoubleSpinBox();
	teneson->setRange(0, 100);
	teneson->setDecimals(2);

	controls.push_back(qMakePair(1, teneson));

	auto *rbf_type = new QComboBox();
	rbf_type->addItems
	(
		{
			QStringLiteral("Gaussian exp(-r^2/b^2)"),
			QStringLiteral("Quadriga sqrt(1+r2/b)"),
			QStringLiteral("Inv.quadriga 1/sqrt(1+r2/b)"),
			QStringLiteral("Polyharmonic spline")
		}
	);
	controls.push_back(qMakePair(2, rbf_type));

	auto *rbf_param = new QDoubleSpinBox();
	rbf_param->setRange(-100, 100);
	rbf_param->setValue(2);
	rbf_param->setDecimals(3);
	controls.push_back(qMakePair(2, rbf_param));

	auto *kriging_type = new QComboBox();
	kriging_type->addItems
	(
		{
			QStringLiteral("exp.variogram ñ0+ñ1(1-exp(-r/c2))"),
			QStringLiteral("gauss.variogram ñ0+ñ1(1-exp(-r^2/c2))"),
			QStringLiteral("spher.variogram ñ0+ñ1(3r/2c2-1/2(r/c2)^3)")
		}
	);
	controls.push_back(qMakePair(3, kriging_type));

	auto *kriging_auto= new QCheckBox(QStringLiteral("auto params"));
	kriging_auto->setChecked(true);
	controls.push_back(qMakePair(3, kriging_auto));

	auto *kriging_param_0 = new QDoubleSpinBox();
	kriging_param_0->setRange(-100, 100);
	kriging_param_0->setEnabled(false);
	kriging_param_0->setValue(1);
	kriging_param_0->setDecimals(3);
	controls.push_back(qMakePair(3, kriging_param_0));

	auto *kriging_param_1 = new QDoubleSpinBox();
	kriging_param_1->setRange(-100, 100);
	kriging_param_1->setEnabled(false);
	kriging_param_1->setValue(1);
	kriging_param_1->setDecimals(3);
	controls.push_back(qMakePair(3, kriging_param_1));

	auto *kriging_param_2 = new QDoubleSpinBox();
	kriging_param_2->setRange(-100, 100);
	kriging_param_2->setEnabled(false);
	kriging_param_2->setValue(1);
	kriging_param_2->setDecimals(3);
	controls.push_back(qMakePair(3, kriging_param_2));

    auto *colorGroupBox = new QGroupBox(QStringLiteral("Custom gradient"));

    QLinearGradient grBtoY(0, 0, 1, 100);
    grBtoY.setColorAt(1.0, Qt::black);
    grBtoY.setColorAt(0.67, Qt::blue);
    grBtoY.setColorAt(0.33, Qt::red);
    grBtoY.setColorAt(0.0, Qt::yellow);
    QPixmap pm(24, 100);
    QPainter pmp(&pm);
    pmp.setBrush(QBrush(grBtoY));
    pmp.setPen(Qt::NoPen);
    pmp.drawRect(0, 0, 24, 100);
    QPushButton *gradientBtoYPB = new QPushButton(widget);
    gradientBtoYPB->setIcon(QIcon(pm));
    gradientBtoYPB->setIconSize(QSize(24, 100));

    QLinearGradient grGtoR(0, 0, 1, 100);
    grGtoR.setColorAt(1.0, Qt::darkGreen);
    grGtoR.setColorAt(0.5, Qt::yellow);
    grGtoR.setColorAt(0.2, Qt::red);
    grGtoR.setColorAt(0.0, Qt::darkRed);
    pmp.setBrush(QBrush(grGtoR));
    pmp.drawRect(0, 0, 24, 100);
	auto *gradientGtoRPB = new QPushButton(widget);
    gradientGtoRPB->setIcon(QIcon(pm));
    gradientGtoRPB->setIconSize(QSize(24, 100));

    auto *colorHBox = new QHBoxLayout;
    colorHBox->addWidget(gradientBtoYPB);
    colorHBox->addWidget(gradientGtoRPB);
    colorGroupBox->setLayout(colorHBox);

 
    vLayout->addWidget(new QLabel(QStringLiteral("calc method")));
    vLayout->addWidget(calc_type);
	
	auto *l = new QLabel(QStringLiteral("tension"));
	vLayout->addWidget(l);
	vLayout->addWidget(teneson);
	controls.push_back(qMakePair(1, l));

	vLayout->addWidget(l=new QLabel(QStringLiteral("rbf function")));
	vLayout->addWidget(rbf_type);
	controls.push_back(qMakePair(2, l));

	vLayout->addWidget(l=new QLabel(QStringLiteral("rbf param")));
	vLayout->addWidget(rbf_param);
	controls.push_back(qMakePair(2, l));

	vLayout->addWidget(l=new QLabel(QStringLiteral("krigin function")));
	vLayout->addWidget(kriging_type);
	controls.push_back(qMakePair(3, l));
	
	vLayout->addWidget(kriging_auto);

	vLayout->addWidget(l=new QLabel(QStringLiteral("C0")));
	controls.push_back(qMakePair(3, l));
	vLayout->addWidget(kriging_param_0);
	vLayout->addWidget(l=new QLabel(QStringLiteral("C1")));
	controls.push_back(qMakePair(3, l));
	vLayout->addWidget(kriging_param_1);
	vLayout->addWidget(l= new QLabel(QStringLiteral("C2")));
	vLayout->addWidget(kriging_param_2);
	controls.push_back(qMakePair(3, l));

    vLayout->addWidget(colorGroupBox);
	auto *bCalc = new QPushButton(QStringLiteral("Calculate"));
	vLayout->addWidget(bCalc);


    widget->show();

    auto *modifier = new SurfaceGraph(graph);

	const auto set_calcul_type=[modifier, &controls](int val)
	{
		std::for_each(controls.begin(), controls.end(), [val](auto &v) {v.second->setVisible(val == v.first); });
		modifier->setCalculType(val);
	};

	set_calcul_type(0);

	QObject::connect(bCalc, &QPushButton::pressed, modifier, &SurfaceGraph::calculate);
    
	QObject::connect(teneson, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), modifier, &SurfaceGraph::setTension);

	QObject::connect(calc_type, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), modifier, set_calcul_type);
	QObject::connect(rbf_type, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), modifier, &SurfaceGraph::setRBFType);
	QObject::connect(rbf_param, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), modifier, &SurfaceGraph::setRBFParam);

	QObject::connect(kriging_type, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), modifier, &SurfaceGraph::setKrigingType);
	QObject::connect(kriging_auto, &QCheckBox::clicked, modifier, [&](bool val)
	{
		kriging_param_0->setEnabled(!val);
		kriging_param_1->setEnabled(!val);
		kriging_param_2->setEnabled(!val);
		modifier->setKrigingAuto(val);
	}
	);
	
		
	QObject::connect(kriging_param_0, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), modifier, &SurfaceGraph::setKrigingParam<0>);
	QObject::connect(kriging_param_1, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), modifier, &SurfaceGraph::setKrigingParam<1>);
	QObject::connect(kriging_param_2, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), modifier, &SurfaceGraph::setKrigingParam<2>);

	QObject::connect(gradientBtoYPB, &QPushButton::pressed,modifier, &SurfaceGraph::setBlackToYellowGradient);
    QObject::connect(gradientGtoRPB, &QPushButton::pressed,modifier, &SurfaceGraph::setGreenToRedGradient);

    return app.exec();
}
