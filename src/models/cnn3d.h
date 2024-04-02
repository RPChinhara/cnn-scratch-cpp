#pragma once

class CNN3D
{
  public:
    void Train();
    void Predict();

  private:
    std::vector<Ten> ForwardPropagation(const Ten &input, const std::vector<Ten> &kernel, const size_t stride);
};