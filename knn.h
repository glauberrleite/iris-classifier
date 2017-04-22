#ifndef KNN_H
#define KNN_H

#include "iris.h"
#include "supervised-learning.h"
#include "attribute-type.h"
#include <vector>
#include <cmath>

class KNN : public SupervisedLearning {
public:
	KNN(const std::vector<Iris*> &trainingData, unsigned int k = 5, bool normalization = false);
	int classificate(float sepalLength, float sepalWidth, float petalLength, float petalWidth);
private:
	unsigned int k;
	std::vector<Iris*> trainingData;
	void findMaxMinAttributes();
	float normalize(float attribute, AttributeType type);
	bool normalization;
	float maxSepalLength;
	float minSepalLength;
	float maxSepalWidth;
	float minSepalWidth;
	float maxPetalLength;
	float minPetalLength;
	float maxPetalWidth;
	float minPetalWidth;
};

#endif
