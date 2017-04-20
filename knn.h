#ifndef KNN_H
#define KNN_H

#include"iris.h"
#include<vector>
#include<cmath>

class KNN {
public:
	KNN(const std::vector<Iris*> &trainingData, unsigned int k = 5);
	int classificate(float sepalLength, float sepalWidth, float petalLength, float petalWidth);
private:
	unsigned int k;
	std::vector<Iris*> trainingData;
};

#endif
