#pragma once

class CNN3D
{
  public:
    void Train();
    void Predict();

  private:
    std::vector<Tensor> ForwardPropagation(const Tensor &input, const std::vector<Tensor> &kernel, const size_t stride);
};