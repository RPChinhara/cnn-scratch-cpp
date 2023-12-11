#pragma once

#include <iostream>
#include <vector>

class Tensor
{
public:
    Tensor() = default;
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
    float& operator[](const size_t idx) const;
    friend Tensor operator-(const float sca, const Tensor& o);
    friend Tensor operator*(const float sca, const Tensor& o);
    friend std::ostream& operator<<(std::ostream& os, const Tensor& o);

    size_t size;
    std::vector<size_t> shape;
    float *elem = nullptr;
    size_t num_ch_dim;
};