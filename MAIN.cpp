

#include<iostream>
#include<string>
#include<vector>
using namespace std;

#include "calculation.h"

int main() {

	calculate asd;
	asd.initialize();
	vector<double> a = { 0.5,0.5 };   // input datas


	for (size_t i = 0; i < 10000; i++)
	{
		asd.forward(a);
		asd.backword(a, 10);   // desire output data is 10
	}

	system("pause");
	return 0;
}














