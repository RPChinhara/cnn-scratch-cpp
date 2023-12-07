#pragma once

#include <iostream>
#include <vector>

class Tensor
{
public:
    Tensor() = default;
    Tensor(const std::vector<float> elem, const std::vector<unsigned int> shape);
    Tensor(const Tensor& o);
    Tensor(Tensor&& o) noexcept;
    ~Tensor();
    Tensor operator+(const Tensor& o) const;
    Tensor operator-(const Tensor& o) const;
    Tensor operator*(const Tensor& o) const;
    Tensor operator/(const Tensor& o) const;
    Tensor& operator=(const Tensor& o);
    Tensor operator+=(const Tensor& o) const;
    Tensor operator-=(const Tensor& o) const;
    float& operator[](const unsigned int idx) const;
    friend Tensor operator-(const float sca, const Tensor& o);
    friend Tensor operator*(const float sca, const Tensor& o);
    friend std::ostream& operator<<(std::ostream& os, const Tensor& o);

    float *elem = nullptr;
    unsigned int num_ch_dim;
    std::vector<unsigned int> shape;
    unsigned int size;
};