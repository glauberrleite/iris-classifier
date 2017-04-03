#ifndef MLP_H
#define MLP_H

#include"iris.h"
#include<vector>
#include<cmath>

class MLP {
public:
  MLP(int numberOfHiddenLayers = 1, int numberOfHiddenNeurons = 4, int numberOfClasses = 3);
  void train(const std::vector<Iris*> &data);
  int classificate(float sepalLength, float sepalWidth, float petalLength, float petalWidth);
private:
  void buildNetwork(float sepalLength, float sepalWidth, float petalLength, float petalWidth);
  int numberOfHiddenLayers;
  int numberOfHiddenNeurons;
  float ** hiddenNeurons;
  std::vector<float> outputs;
  float *** weights;
  float sigmoid(float x);
};

#endif
