#include "knn.h"

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

void KNN::findMaxMinAttributes() {
	maxSepalLength = maxSepalWidth = maxPetalLength = maxPetalWidth = 0;
	minSepalLength = minSepalWidth = minPetalLength = minPetalWidth = 99.9;

	for (Iris * sample : this->trainingData) {
		maxSepalLength = sample->getSepalLength() > maxSepalLength ? sample->getSepalLength() : maxSepalLength;
		minSepalLength = sample->getSepalLength() < minSepalLength ? sample->getSepalLength() : minSepalLength;

		maxSepalWidth = sample->getSepalWidth() > maxSepalWidth ? sample->getSepalWidth() : maxSepalWidth;
		minSepalWidth = sample->getSepalWidth() < minSepalWidth ? sample->getSepalWidth() : minSepalWidth;

		maxPetalLength = sample->getPetalLength() > maxPetalLength ? sample->getPetalLength() : maxPetalLength;
		minPetalLength = sample->getPetalLength() < minPetalLength ? sample->getPetalLength() : minPetalLength;

		maxPetalWidth = sample->getPetalWidth() > maxPetalWidth ? sample->getPetalWidth() : maxPetalWidth;
		minPetalWidth = sample->getPetalWidth() < minPetalWidth ? sample->getPetalWidth() : minPetalWidth;
	}
}

float KNN::normalize(float attribute, AttributeType type) {
	float result = 0;

	switch (type) {
		case SL: result = (attribute - this->minSepalLength)/(this->maxSepalLength - this->minSepalLength);
							break;
		case SW: result = (attribute - this->minSepalWidth)/(this->maxSepalWidth - this->minSepalWidth);
							break;
		case PL: result = (attribute - this->minPetalLength)/(this->maxPetalLength - this->minPetalLength);
							break;
		case PW: result = (attribute - this->minPetalWidth)/(this->maxPetalWidth - this->minPetalWidth);
							break;
	}

	return result;
}

KNN::KNN(const std::vector<Iris*> &trainingData, unsigned int k, bool normalization){
	this->trainingData = trainingData;
	this->k = k;

	this->normalization = normalization;

	if (normalization)
		findMaxMinAttributes();
}

int KNN::classificate(float sepalLength, float sepalWidth, float petalLength, float petalWidth){
	int nearestNeighbors[this->k][2];

	// Initializing nearestNeighbors
	for (int i = 0; i < k; ++i) {
		nearestNeighbors[i][0] = -1;
		nearestNeighbors[i][1] = 99.9;
	}

	for (Iris * sample : this->trainingData) {
			// Calculating distance from training sample to given values
			float diffSL, diffSW, diffPL, diffPW;
			if (normalization) {
				diffSL = normalize(sample->getSepalLength(), AttributeType::SL) - normalize(sepalLength, AttributeType::SL);
				diffSW = normalize(sample->getSepalWidth(), AttributeType::SW) - normalize(sepalWidth, AttributeType::SW);
				diffPL = normalize(sample->getPetalLength(), AttributeType::PL) - normalize(petalLength, AttributeType::PL);
				diffPW = normalize(sample->getPetalWidth(), AttributeType::PW) - normalize(petalWidth, AttributeType::PW);
			} else {
				diffSL = sample->getSepalLength() - sepalLength;
				diffSW = sample->getSepalWidth() - sepalWidth;
				diffPL = sample->getPetalLength() - petalLength;
				diffPW = sample->getPetalWidth() - petalWidth;
			}
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
