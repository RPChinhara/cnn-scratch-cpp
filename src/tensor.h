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
    float& operator[](const unsigned int ind) const;
    friend Tensor operator-(const float sca, const Tensor& o);
    friend Tensor operator*(const float sca, const Tensor& o);
    friend std::ostream& operator<<(std::ostream& os, const Tensor& o);
    Tensor T() const;

    float                    *_elem = nullptr;
    unsigned int              _num_ch_dim;
    std::vector<unsigned int> _shape;
    unsigned int              _size;
};