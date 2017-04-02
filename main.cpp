#include<iostream>
#include<iomanip>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<cstdlib>
#include"iris.h"
#include"mlp.h"

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

/*
  // Test
  for (Iris * iris : trainingData) {
    cout << std::setprecision(2) << std::fixed << iris->getSepalLength();
    cout << " , ";
    cout << std::setprecision(2) << std::fixed << iris->getSepalWidth();
    cout << " , ";
    cout << std::setprecision(2) << std::fixed << iris->getPetalLength();
    cout << " , ";
    cout << std::setprecision(2) << std::fixed << iris->getPetalWidth();
    cout << " , ";
    cout << iris->getType() << endl;
  }
*/

  // Run MLP
  MLP mlp;

  cout << "Training MLP..." << endl;
  mlp.train(trainingData);

  cout << "Testing" << endl;
  int counter = 1;
  for (Iris* iris : testingData) {
    int estimative = mlp.classificate(iris->getSepalLength(), iris->getSepalWidth(),
    iris->getPetalLength(), iris->getPetalWidth());

    cout << "Test " << counter << ": ";
    cout << "Estimative -> " << estimative << " ";
    cout << "Data -> " << iris->getType() << " ";

    if (estimative == iris->getType()) {
      cout << "(OK)";
    } else {
      cout << "(FAIL)";
    }

    cout << endl;

    counter++;
  }

  return 0;
}
