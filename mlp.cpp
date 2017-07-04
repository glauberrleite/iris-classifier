#include "mlp.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>

float MLP::sigmoid(float x) {
  return 1/(1 + exp(-x));
}

float MLP::derivative_sigmoid(float x) {
  return exp(x)/pow((1 + exp(x)), 2);
}

MLP::MLP(int numberOfHiddenLayers, int numberOfHiddenNeurons, float learningRate) {
  this->numberOfHiddenLayers = numberOfHiddenLayers;
  this->numberOfHiddenNeurons = numberOfHiddenNeurons;

  this->hiddenNeurons = new float*[numberOfHiddenLayers];
  for (int i = 0; i < numberOfHiddenLayers; i++)
    this->hiddenNeurons[i] = new float[numberOfHiddenNeurons];

  // As we have 3 possible classes, there are 3 outputs
  this->outputs.resize(3);

  // Pseudo Random generator
  std::mt19937 randomGenerator;
  std::uniform_int_distribution<std::mt19937::result_type> dist(0,100);
  randomGenerator.seed(std::random_device()());

  /* The anatomy of weights
  // weights[i][j][k] where:
  // i is the layer
  // j is the output neuron
  // k is the input neuron
  */


  // Building weights matrix and filling it with random values
  this->weights = new float**[numberOfHiddenLayers + 1];

  // In the first layer we have 4 weights, for the four variables
  this->weights[0] = new float*[numberOfHiddenNeurons];
  for (int j = 0; j < numberOfHiddenNeurons; ++j){
    this->weights[0][j] = new float[4];

    for (int k = 0; k < 4; ++k)
      this->weights[0][j][k] = ((float) dist(randomGenerator))/1000;
  }


  // The rest of the layers, except output, use the layer before as input
  for (int i = 1; i < numberOfHiddenLayers; ++i) {
    this->weights[i] = new float*[numberOfHiddenNeurons];

    for (int j = 0; j < numberOfHiddenNeurons; ++j){
      this->weights[i][j] = new float[numberOfHiddenNeurons];

      for (int k = 0; k < numberOfHiddenNeurons; ++k)
        this->weights[i][j][k] = ((float) dist(randomGenerator))/1000;
    }
  }

  // The last layer is the layer of outputs
  this->weights[numberOfHiddenLayers] = new float*[outputs.size()];
  for (int j = 0; j < outputs.size(); ++j) {
    this->weights[numberOfHiddenLayers][j] = new float[numberOfHiddenNeurons];

    for(int k = 0; k < numberOfHiddenNeurons; ++k)
      this->weights[numberOfHiddenLayers][j][k] = ((float) dist(randomGenerator))/1000;

  }

  this->learningRate = learningRate;

}

void MLP::buildNetwork(float sepalLength, float sepalWidth, float petalLength, float petalWidth) {
  // For the first layer
  for (int j = 0; j < this->numberOfHiddenNeurons; j++) {
    this->hiddenNeurons[0][j] = 0;

    // Calculating first layer neuron weights, we know how much inputs we have
    this->hiddenNeurons[0][j] += sepalLength * this->weights[0][j][0];
    this->hiddenNeurons[0][j] += sepalWidth * this->weights[0][j][1];
    this->hiddenNeurons[0][j] += petalLength * this->weights[0][j][2];
    this->hiddenNeurons[0][j] += petalWidth * this->weights[0][j][3];

    // Passing to the activation function, in this case, the logistic function
    this->hiddenNeurons[0][j] = sigmoid(this->hiddenNeurons[0][j]);
  }

  // For extra layers, if there's any
  for (int i = 1; i < this->numberOfHiddenLayers; i++) {
    for (int j = 0; j < this->numberOfHiddenNeurons; j++) {
      this->hiddenNeurons[i][j] = 0;

      // This time, each k-th neuron from the antecessor layer (i -1) is a input
      for (int k = 0; k < this->numberOfHiddenNeurons; k++)
        this->hiddenNeurons[i][j] += this->hiddenNeurons[i - 1][k] * this->weights[i][j][k];

      // Passing to the activation function, in this case, the logistic function
      this->hiddenNeurons[i][j] = sigmoid(this->hiddenNeurons[i][j]);
    }
  }

  // Now, the output vector gets the last hidden neurons as input
  for (int i = 0; i < this->outputs.size(); i++){
    this->outputs[i] = 0;

    for (int j = 0; j < this->numberOfHiddenNeurons; j++){
      this->outputs[i] += this->hiddenNeurons[this->numberOfHiddenLayers - 1][j] * this->weights[this->numberOfHiddenLayers][i][j];
    }

    this->outputs[i] = sigmoid(this->outputs[i]);
  }
}

void MLP::train(const std::vector<Iris*> &data, int numberOfEpochs){

  for (int epoch = 1; epoch <= numberOfEpochs; epoch++) {
    float outputDelta[this->outputs.size()];
    float hiddenDelta[this->numberOfHiddenLayers][this->numberOfHiddenNeurons];

    int counter = 0;
    for (Iris * iris : data) {
    	counter++;
    	std::cout << "---------" << std::endl;
    	std::cout << "Training " << counter << std::endl;

      int estimative = classificate(iris->getSepalLength(), iris->getSepalWidth(),
        iris->getPetalLength(), iris->getPetalWidth());

      if(estimative == iris->getType())
        continue;

      // propagation starts by output
      for (int i = 0; i < outputs.size(); i++) {
  	    float error = 0.0;
        if (i == iris->getType()) {

          error = (1 - outputs[i]);
          outputDelta[i] = error * derivative_sigmoid(outputs[i]);

        } else {

          error = (0 - outputs[i]);
          outputDelta[i] = error * derivative_sigmoid(outputs[i]);

        }
      }

      // propagation goes through the last hiddenLayer
      for (int i = 0; i < numberOfHiddenNeurons; i++) {
        hiddenDelta[numberOfHiddenLayers - 1][i] = 0;

        for (int j = 0; j < outputs.size(); j++) {
            hiddenDelta[numberOfHiddenLayers - 1][i] += outputDelta[j] * weights[numberOfHiddenLayers][j][i];
        }

        hiddenDelta[numberOfHiddenLayers - 1][i] *= derivative_sigmoid(hiddenNeurons[numberOfHiddenLayers - 1][i]);
      }

      // propagation through layer to layer
      for (int i = numberOfHiddenLayers - 2; i >= 0; i--) {

        for (int j = 0; j < numberOfHiddenNeurons; j++) {
          hiddenDelta[i][j] = 0;

          for (int k = 0; k < numberOfHiddenNeurons; k++) {
            hiddenDelta[i][j] += hiddenDelta[i + 1][k] * weights[i + 1][k][j];
          }

          hiddenDelta[i][j] *= derivative_sigmoid(hiddenNeurons[i][j]);
        }
      }

      // Weight adjustment
      // Starting from the inputs
      for (int j = 0; j < numberOfHiddenNeurons; j++) {
        weights[0][j][0] += learningRate * hiddenDelta[0][j] * iris->getSepalLength();
        weights[0][j][1] += learningRate * hiddenDelta[0][j] * iris->getSepalWidth();
        weights[0][j][2] += learningRate * hiddenDelta[0][j] * iris->getPetalLength();
        weights[0][j][3] += learningRate * hiddenDelta[0][j] * iris->getPetalWidth();
      }

      // For the hidden layers weights
      for (int i = 1; i < numberOfHiddenLayers; i++) {
        for (int j = 0; j < numberOfHiddenNeurons; j++) {
          for (int k = 0; k < numberOfHiddenNeurons; k++)
            weights[i][j][k] += learningRate * hiddenDelta[i][j] * hiddenNeurons[i - 1][k];
        }
      }

      // For the output layer
      for (int i = 0; i < outputs.size(); i++)
        for (int j = 0; j < numberOfHiddenNeurons; j++)
          weights[numberOfHiddenLayers][i][j] += learningRate * outputDelta[i] * hiddenNeurons[numberOfHiddenLayers - 1][j];

    }
  }
}

int MLP::classificate(float sepalLength, float sepalWidth, float petalLength, float petalWidth){
  buildNetwork(sepalLength, sepalWidth, petalLength, petalWidth);
  int max = 0;
  int result = 0;

  std::cout << "------------" << std::endl;
  for (int i = 0; i < this->outputs.size(); i++){
    std::cout << "output[" << i << "] = " << outputs[i] << std::endl;
    if(this->outputs[i] > max){
      max = this->outputs[i];
      result = i;
    }
  }

  return result;
}
