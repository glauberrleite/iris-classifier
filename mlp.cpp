#include"mlp.h"

MLP::MLP(){

}

void MLP::train(const std::vector<Iris*> &data){

}

int MLP::classificate(float sepalLength, float sepalWidth, float petalLength, float petalWidth){
  return 0;
}

float MLP::sigmoid(float x){
  return 1/(1 + pow(M_E, -x));
}
