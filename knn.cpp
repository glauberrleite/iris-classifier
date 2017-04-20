#include"knn.h"

KNN::KNN(const std::vector<Iris*> &trainingData, unsigned int k){
	this->trainingData = trainingData;
	this->k = k;
}

int KNN:classificate(float sepalLength, float sepalWidth, float petalLength, float petalWidth){
	int nearestNeighbors[this->k][2];
	
	// Initializing nearestNeighbors
	for (int i = 0; i < k; ++i) {
		nearestNeighbors[i][0] = -1;
		nearestNeighbors[i][1] = FLOAT_MAX;
	}
	
	
	for (Iris * sample : trainingData) {
			float diffSL = sample->getSepalLength() - sepalLength;
			float diffSW = sample->getSepalWidth() - sepalWidth;
			float diffPL = sample->getPetalLength() - petalLength;
			float diffPW = sample->getPetalWidth() - petalWidth;
			float distance = sqrt(pow(diffSL, 2) + pow(diffSW, 2) + pow(diffPL, 2) + pow(diffPW, 2));
			
			for (int i = 0; i < k; ++i) {
				
			}
	}
	
	// Creating counter for each one of the three types of Iris
	int counter[3];
	
	for (int i = 0; i < 3; i++)
		counter[i] = 0;
	
	for (int i = 0; i < k; ++i) {
		int type = nearestNeighbors[i][0];
		counter[type]++;
	}
	
	// Finding the most repeating type in the neighbor
	int estimative = -1;
	
	for (int i = 0; i < 3; i++){
		bool max = true;
		
		for (int j = 0; j < 3; j++)
			if(counter[i] < counter[j]){
				max = false;
				continue;
			}
		
		if (max) {
			estimative = i;
			break;
		}
	}
	
	return estimative;
}
