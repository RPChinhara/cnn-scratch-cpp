#pragma once

class CNN1D
{
  public:
    void Train();
    void Predict();

  private:
    std::vector<ten> ForwardPropagation(const ten &input, const std::vector<ten> &kernel, const size_t stride);
};