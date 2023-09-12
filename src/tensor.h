#pragma once

#include "types.h"

#include <iostream>
#include <vector>

class Tensor {
public:
    Tensor() = default;
    Tensor(const std::vector<f32> elem, const std::vector<u32> shape);
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
    f32& operator[](const u32 ind) const;
    friend Tensor operator-(const f32 sca, const Tensor& o);
    friend Tensor operator*(const f32 sca, const Tensor& o);
    friend std::ostream& operator<<(std::ostream& os, const Tensor& o);
    Tensor T() const;

    f32             *_elem = nullptr;
    u32              _num_ch_dim;
    std::vector<u32> _shape;
    u32              _size;
};