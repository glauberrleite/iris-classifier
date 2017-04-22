#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include "iris.h"
#include "supervised-learning.h"
#include "method.h"
#include "mlp.h"
#include "knn.h"

using namespace std;

void readFile(const string filePath, vector<Iris*> &data){
  ifstream dataFile;
  dataFile.open(filePath.c_str());

  while(!dataFile.eof()){

    // Reading new line and building stream
    string line;

    if(!getline(dataFile, line))
      break; // avoid empty lines at the end
    stringstream lineStream(line);

    // Reading each cell
    string cell[5];

    for(int i = 0; i < 5; ++i){
      getline(lineStream, cell[i], ',');
    }

    float sepalLength = strtof(cell[0].c_str(), 0);
    float sepalWidth = strtof(cell[1].c_str(), 0);
    float petalLength = strtof(cell[2].c_str(), 0);
    float petalWidth = strtof(cell[3].c_str(), 0);
    string typeName = cell[4];

    // Building object and pushing it to vector
    data.push_back(new Iris(sepalLength, sepalWidth, petalLength, petalWidth, typeName));
  }

  dataFile.close();

}

Method getMethodChoice(){
  cout << "1 - MLP" << endl;
  cout << "2 - KNN" << endl;

  int choice;
  cin >> choice;

  if (choice < 1 || choice > 2) {
    cout << "Choose a valid option" << endl;
    return getMethodChoice();
  }

  return static_cast<Method>(choice);
}

int main(){

  cout << "Iris Classifier" << endl;

  vector<Iris*> trainingData;
  vector<Iris*> testingData;

  // Reading file with training values
  cout << "Reading iris-training.data file..." << endl;

  readFile("dataset/iris-training.data", trainingData);

  cout << ">> There are " << trainingData.size() << " instances for training" << endl;

  // Reading file with testing values
  cout << "Reading iris-testing.data file..." << endl;

  readFile("dataset/iris-testing.data", testingData);

  cout << ">> There are " << testingData.size() << " instances for testing" << endl;

  cout << "Choose algorithm: " << endl;
  Method method = getMethodChoice();

  SupervisedLearning * algorithm;

  switch (method) {
	  case M_MLP: {
		  cout << "Using Multilayer Perceptron Algorithm" << endl;

		  // Run MLP
		  algorithm = new MLP(2, 4, 0.1);

		  cout << "Training MLP..." << endl;
		  static_cast<MLP*>(algorithm)->train(trainingData);

		  break;
	  }
	  case M_KNN: {
		  cout << "Using K Nearest Neighbor" << endl;

      int k = 0;
      bool valid = false;
      bool normalize;

      while (!valid) {
        cout << "Set k: ";
        cin >> k;

        if (k >= 1 && k <= trainingData.size()) {
          valid = true;
        } else {
          cout << "Invalid k number!" << endl;
          continue;
        }

        cout << "Normalize data [0/1]? ";
        cin >> normalize;

        normalize = normalize > 1 || normalize < 0 ? 1 : normalize;
      }

		  algorithm = new KNN(trainingData, k, normalize);

      break;
	  }
  }

  // Initializing Confusion Matrix
  int confMatrix[3][3];

  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      confMatrix[i][j] = 0;

  // Testing
  cout << "Testing" << endl;

  int counter = 1;
  int hitCounter;
  hitCounter = 0;

  for (Iris* iris : testingData) {

    int estimative = algorithm->classificate(iris->getSepalLength(), iris->getSepalWidth(),
      iris->getPetalLength(), iris->getPetalWidth());

    cout << "Test " << counter << ": ";
    cout << "Estimative -> " << estimative << " ";
    cout << "Data -> " << iris->getType() << " ";

    confMatrix[estimative][iris->getType()]++;

    if (estimative == iris->getType()) {
      cout << "(OK)";
      hitCounter++;
    } else {
      cout << "(FAIL)";
    }

    cout << endl;

    counter++;
  }

  // Writing confusion Matrix
  cout << "Confusion Matrix" << endl;
  cout << "|-----------|-------|-------|-------|" << endl;
  cout << "|Est....Real|   0   |   1   |   2   |" << endl;
  cout << "|-----------|-------|-------|-------|" << endl;

  for (int i = 0; i < 3; i++) {
      cout << "|     " + to_string(i) + "     |";

      for (int j = 0; j < 3; j++) {
          cout << "   " + to_string(confMatrix[i][j]) + "  ";

          if ((confMatrix[i][j] / 10) == 0)
            cout << " ";

          cout << "|" ;
      }

      cout << endl;

      cout << "|-----------|-------|-------|-------|" << endl;
  }

  float hitRate = (static_cast<float>(hitCounter) / testingData.size()) * 100;

  cout << "Hit Rate: " << fixed << setprecision(2) << hitRate << "%" << endl;


  return 0;
}
