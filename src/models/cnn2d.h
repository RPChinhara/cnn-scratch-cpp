#pragma once

#include <vector>

class Ten;

class CNN2D
{
  public:
    CNN2D(const std::vector<size_t> &filters, float const learning_rate);
    void Train(const Ten &xTrain, const Ten &yTrain, const Ten &xVal, const Ten &yVal);
    void Predict(const Ten &xTest, const Ten &yTest);

  private:
    std::vector<Ten> ForwardPropagation(const Ten &input, const std::vector<Ten> &kernel, const size_t stride);

    std::vector<size_t> filters;
    float learning_rate;
};