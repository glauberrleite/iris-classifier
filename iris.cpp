#include"iris.h"

void Iris::init(float sepalLength, float sepalWidth, float petalLength, float petalWidth){
  this->sepalLength = sepalLength;
  this->sepalWidth = sepalWidth;
  this->petalLength = petalLength;
  this->petalWidth = petalWidth;
}

Iris::Iris(float sepalLength, float sepalWidth, float petalLength, float petalWidth, int type){
  init(sepalLength, sepalWidth, petalLength, petalWidth);

  // As we have only 3 types of iris, if we get inconsistence, let it be 0
  this->type = (type > 2) && (type < 0) ? 0 : type;
}

Iris::Iris(float sepalLength, float sepalWidth, float petalLength, float petalWidth, std::string typeName){

  init(sepalLength, sepalWidth, petalLength, petalWidth);

  // By default, we have Iris-setosa
  this->type = 0;

  // We need to be aware of windows carriage return
  if (typeName == "Iris-versicolor" || typeName == "Iris-versicolor\r")
    this->type = 1;
  else if (typeName == "Iris-virginica" || typeName== "Iris-virginica\r")
    this->type = 2;
}

float Iris::getSepalLength(){
  return sepalLength;
}

float Iris::getSepalWidth(){
  return sepalWidth;
}

float Iris::getPetalLength(){
  return petalLength;
}

float Iris::getPetalWidth(){
  return petalWidth;
}

int Iris::getType(){
  return type;
}
