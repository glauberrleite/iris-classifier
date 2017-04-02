#include<string>

class Iris {
public:
  Iris(float sepalLength, float sepalWidth, float petalLength, float petalWidth, int type);
  Iris(float sepalLength, float sepalWidth, float petalLength, float petalWidth, std::string typeName);

  float getSepalLength();
  float getSepalWidth();
  float getPetalLength();
  float getPetalWidth();
  int getType();
private:
  void init(float sepalLength, float sepalWidth, float petalLength, float petalWidth);
  float sepalLength;
  float sepalWidth;
  float petalLength;
  float petalWidth;
  int type;
};
