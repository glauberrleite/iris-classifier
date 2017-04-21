#include"knn.h"

int farest (int (*nearestNeighbors)[2], unsigned int k) {
	for (int i = 0; i < k; i++) {
		bool max = true;
		for (int j = 0; j < k; j++)
			if(nearestNeighbors[i][1] < nearestNeighbors[j][1])
				max = false;

		if (max) {
			return i;
		}
	}
}

KNN::KNN(const std::vector<Iris*> &trainingData, unsigned int k){
	this->trainingData = trainingData;
	this->k = k;
}

int KNN::classificate(float sepalLength, float sepalWidth, float petalLength, float petalWidth){
	int nearestNeighbors[this->k][2];

	// Initializing nearestNeighbors
	for (int i = 0; i < k; ++i) {
		nearestNeighbors[i][0] = -1;
		nearestNeighbors[i][1] = 99999.9;
	}

	for (Iris * sample : trainingData) {
			// Calculating distance from training sample to given sample
			float diffSL = sample->getSepalLength() - sepalLength;
			float diffSW = sample->getSepalWidth() - sepalWidth;
			float diffPL = sample->getPetalLength() - petalLength;
			float diffPW = sample->getPetalWidth() - petalWidth;
			float distance = sqrt(pow(diffSL, 2) + pow(diffSW, 2) + pow(diffPL, 2) + pow(diffPW, 2));

			for (int i = 0; i < k; ++i) {
				if (nearestNeighbors[i][0] == -1) {
					nearestNeighbors[i][0] = sample->getType();
					nearestNeighbors[i][1] = distance;

					break;
				}

				if (nearestNeighbors[i][1] > distance) {
					// Removing the farest from the nearest neighbor
					int index = farest(nearestNeighbors, k);
					nearestNeighbors[index][0] = sample->getType();
					nearestNeighbors[index][1] = distance;

					break;
				}
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

	// Finding the most repeating type in the neighborhood
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
