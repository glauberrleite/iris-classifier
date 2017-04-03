#include"mlp.h"
#include<random>
#include<algorithm>

float MLP::sigmoid(float x) {
  return 1/(1 + exp(-x));
}

MLP::MLP(int numberOfHiddenLayers, int numberOfHiddenNeurons, int numberOfClasses) {
  this->numberOfHiddenLayers = numberOfHiddenLayers;
  this->numberOfHiddenNeurons = numberOfHiddenNeurons;

  this->hiddenNeurons = new float*[numberOfHiddenLayers];
  for (int i = 0; i < numberOfHiddenLayers; i++)
    this->hiddenNeurons[i] = new float[numberOfHiddenNeurons];

  // As we have 3 possible classes, there are 3 outputs
  this->outputs.resize(numberOfClasses);

  // Pseudo Random generator
  std::mt19937 randomGenerator;
  std::uniform_int_distribution<std::mt19937::result_type> dist(0,1);
  randomGenerator.seed(std::random_device()());

  /* The anatomy of weights
  // weights[i][j][k] where:
  // i is the layer
  // j is the output neuron
  // k is the input neuron
  */


  // Building weights matrix and filling it with random values
  this->weights = new float**[numberOfHiddenLayers + 2];

  // In the first layer we have 4 weights, for the four variables
  this->weights[0] = new float*[numberOfHiddenNeurons];
  for (int j = 0; j < numberOfHiddenNeurons; ++j){
    this->weights[0][j] = new float[4];

    for (int k = 0; k < 4; ++k)
      this->weights[0][j][k] = dist(randomGenerator);
  }


  // The rest of the layers, except output, use the layer before as input
  for (int i = 1; i <= numberOfHiddenLayers; ++i) {
    this->weights[i] = new float*[numberOfHiddenNeurons];

    for (int j = 0; j < numberOfHiddenNeurons; ++j){
      this->weights[i][j] = new float[numberOfHiddenNeurons];

      for (int k = 0; k < numberOfHiddenNeurons; ++k)
        this->weights[i][j][k] = dist(randomGenerator);
    }
  }

  // The last layer is the layer of outputs
  this->weights[numberOfHiddenLayers + 1] = new float*[outputs.size()];
  for (int j = 0; j < outputs.size(); ++j) {
    this->weights[numberOfHiddenLayers + 1][j] = new float[numberOfHiddenNeurons];

    for(int k = 0; k < numberOfHiddenNeurons; ++k)
      this->weights[numberOfHiddenLayers + 1][j][k] = dist(randomGenerator);

  }

}

int MLP::classificate(float sepalLength, float sepalWidth, float petalLength, float petalWidth) {
  // For the first layer
  for (int j = 0; j < this->numberOfHiddenNeurons; j++) {
    hiddenNeurons[0][j] = 0;

    // Calculating first layer neuron weights, we know how much inputs we have
    hiddenNeurons[0][j] += sepalLength * weights[0][j][0];
    hiddenNeurons[0][j] += sepalWidth * weights[0][j][1];
    hiddenNeurons[0][j] += petalLength * weights[0][j][2];
    hiddenNeurons[0][j] += petalWidth * weights[0][j][3];

    // Passing to the activation function, in this case, the logistic function
    hiddenNeurons[0][j] = sigmoid(hiddenNeurons[0][j]);
  }

  // For extra layers, if there's any
  for (int i = 1; i < this->numberOfHiddenLayers; i++) {
    for (int j = 0; j < this->numberOfHiddenNeurons; j++) {
      hiddenNeurons[i][j] = 0;

      // This time, each k-th neuron from the antecessor layer (i -1) is a input
      for (int k = 0; k < this->numberOfHiddenNeurons; k++)
        hiddenNeurons[i][j] += hiddenNeurons[i-1][k] * weights[i - 1][j][k];

      // Passing to the activation function, in this case, the logistic function
      hiddenNeurons[i][j] = sigmoid(hiddenNeurons[i][j]);
    }
  }

  // Now, the output vector gets the last hidden neurons as input, and the max value is held
  int max = 0;
  int result = 0;
  for (int i = 0; i < outputs.size(); i++){
    outputs[i] = 0;

    for (int j = 0; j < numberOfHiddenNeurons; j++){
      outputs[i] = hiddenNeurons[numberOfHiddenLayers][j] * weights[numberOfHiddenLayers][i][j];
    }

    outputs[i] = sigmoid(outputs[i]);

    if(outputs[i] > max)
      result = i;
  }

  return result;

}

void MLP::train(const std::vector<Iris*> &data){
  int epoch = 1;

  for (Iris * iris : data){
    //training
  }
}
