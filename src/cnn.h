#pragma once

#include <vector>

class Tensor;

// TODO: I could make parent class called NN?
// NOTE: For one-dimensional data.
class CNN1D
{
  public:
    void Train();
    void Predict();

  private:
    std::vector<Tensor> ForwardPropagation(const Tensor &input, const std::vector<Tensor> &kernel, const size_t stride);
};

// NOTE: For two-dimensional data, such as images (most common).
class CNN2D
{
  public:
    CNN2D(const std::vector<size_t> &filters, float const learning_rate);
    void Train();
    void Predict();

  private:
    std::vector<Tensor> ForwardPropagation(const Tensor &input, const std::vector<Tensor> &kernel, const size_t stride);

    std::vector<size_t> filters;
    float learning_rate;
};

// NOTE: For three-dimensional data, like video or volumetric data.
class CNN3D
{
  public:
    void Train();
    void Predict();

  private:
    std::vector<Tensor> ForwardPropagation(const Tensor &input, const std::vector<Tensor> &kernel, const size_t stride);
};