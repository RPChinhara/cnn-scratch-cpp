#pragma once

#include <vector>

class ten;

class CNN2D
{
  public:
    CNN2D(const std::vector<size_t> &filters, float const learning_rate);
    void Train(const ten &xTrain, const ten &yTrain, const ten &xVal, const ten &yVal);
    void Predict(const ten &xTest, const ten &yTest);

  private:
    std::vector<ten> ForwardPropagation(const ten &input, const std::vector<ten> &kernel, const size_t stride);

    std::vector<size_t> filters;
    float learning_rate;
};