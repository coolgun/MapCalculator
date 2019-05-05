#pragma once
#include<vector>

class solveCrout 
{
public:
	static bool exec	(std::vector<double> &S,std::vector<double> &b)	;
	static bool invert	(int n,std::vector<double> &S);
};


