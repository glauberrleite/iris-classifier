#ifndef MLP_H
#define MLP_H

#include "iris.h"
#include "supervised-learning.h"
#include <vector>
#include <cmath>

class MLP : public SupervisedLearning {
public:
  MLP(int numberOfHiddenLayers = 1, int numberOfHiddenNeurons = 4, float learningRate = 0.1);
  void train(const std::vector<Iris*> &data, int numberOfEpochs = 1);
  int classificate(float sepalLength, float sepalWidth, float petalLength, float petalWidth);
private:
  void buildNetwork(float sepalLength, float sepalWidth, float petalLength, float petalWidth);
  float learningRate;
  int numberOfHiddenLayers;
  int numberOfHiddenNeurons;
  float ** hiddenNeurons;
  std::vector<float> outputs;
  float *** weights;
  float sigmoid(float x);
  float derivative_sigmoid(float x);
};

#endif
