#include<iostream>
#include<iomanip>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>
#include<cstdlib>
#include"iris.h"

using namespace std;

int main(){

  cout << "Iris Classifier" << endl;

  vector<Iris*> data;

  // Reading file
  cout << "Reading iris.data file..." << endl;
  ifstream dataFile;
  dataFile.open("iris.data");

  int counter;
  while(!dataFile.eof()){

    // Reading new line and building stream
    string line;

    if(!getline(dataFile, line))
      break; // avoid empty lines
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

    ++counter;
  }

  cout << "We have " << counter << " instances for training" << endl;

  /*
  // Test
  for (Iris * iris : data) {
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

  return 0;
}
