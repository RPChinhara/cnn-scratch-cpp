#pragma once

#include <iostream>
#include <vector>

class Tensor
{
  public:
    Tensor() = default;
    Tensor(const std::vector<float> elem, const std::vector<size_t> shape);
    ~Tensor();
    Tensor(const Tensor &other);
    Tensor(Tensor &&other);
    Tensor &operator=(const Tensor &other);
    Tensor &operator=(Tensor &&other);
    Tensor operator+(const Tensor &tensor) const;
    Tensor operator-(const Tensor &tensor) const;
    Tensor operator*(const Tensor &tensor) const;
    Tensor operator/(const Tensor &tensor) const;
    Tensor operator+=(const Tensor &other) const;
    Tensor operator-=(const Tensor &other) const;
    float &operator[](const size_t idx) const;
    friend Tensor operator-(const float sca, const Tensor &tensor);
    friend Tensor operator*(const float sca, const Tensor &tensor);
    friend void operator/(const Tensor &tensor, const float sca);
    friend std::ostream &operator<<(std::ostream &os, const Tensor &tensor);

    size_t size;
    std::vector<size_t> shape;
    float *elem = nullptr;
};