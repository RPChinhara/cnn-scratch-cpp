#pragma once

#include <iostream>
#include <vector>

class Tensor
{
public:
    Tensor() = default;
    Tensor(const Tensor& other);
    Tensor(Tensor&& other);
    ~Tensor();
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;
    Tensor& operator=(const Tensor& other);
    Tensor operator+=(const Tensor& other) const;
    Tensor operator-=(const Tensor& other) const;
    float& operator[](const size_t idx) const;
    friend Tensor operator-(const float sca, const Tensor& other);
    friend Tensor operator*(const float sca, const Tensor& other);
    friend std::ostream& operator<<(std::ostream& os, const Tensor& other);

    size_t size;
    std::vector<size_t> shape;
    float *elem = nullptr;
    size_t num_ch_dim;
};