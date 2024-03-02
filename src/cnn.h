#pragma once

#include <vector>

class Tensor;

class CNN1D
{
  public:
    void Train();
    void Predict();

  private:
    std::vector<Tensor> ForwardPropagation(const Tensor &input, const std::vector<Tensor> &kernel, const size_t stride);
};

class CNN2D
{
  public:
    CNN2D(const std::vector<size_t> &filters, float const learning_rate);
    void Train(const Tensor &xTrain, const Tensor &yTrain, const Tensor &xVal, const Tensor &yVal);
    void Predict(const Tensor &xTest, const Tensor &yTest);

  private:
    std::vector<Tensor> ForwardPropagation(const Tensor &input, const std::vector<Tensor> &kernel, const size_t stride);

    std::vector<size_t> filters;
    float learning_rate;
};

class CNN3D
{
  public:
    void Train();
    void Predict();

  private:
    std::vector<Tensor> ForwardPropagation(const Tensor &input, const std::vector<Tensor> &kernel, const size_t stride);
};

class ResNet
{
};