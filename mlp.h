#ifndef MLP_H
#define MLP_H

#include"iris.h"
#include<vector>
#include<cmath>

class MLP {
public:
  MLP();
  void train(const std::vector<Iris*> &data);
  int classificate(float sepalLength, float sepalWidth, float petalLength, float petalWidth);
private:
  float sigmoid(float x);
};

#endif
