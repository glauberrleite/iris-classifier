#ifndef SUPERVISEDLEARNING_H
#define SUPERVISEDLEARNING_H

class SupervisedLearning {
public:
  virtual int classificate(float sepalLength, float sepalWidth, float petalLength, float petalWidth) = 0;
};

#endif
