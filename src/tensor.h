#pragma once

#include <iostream>
#include <vector>

class tensor {
  public:
    size_t size;
    std::vector<size_t> shape;
    float *elems = nullptr;

    tensor() = default;
    tensor(const std::vector<size_t> &shape, const std::vector<float> &elems);
    ~tensor();
    tensor(const tensor &other);
    tensor(tensor &&other) noexcept;
    tensor &operator=(const tensor &other);
    tensor &operator=(tensor &&other) noexcept;

    const std::string get_shape() const;
    tensor &reshape(const std::vector<size_t> &new_shape);

    tensor operator+(const tensor &other) const;
    tensor operator-(const tensor &other) const;
    tensor operator*(const tensor &other) const;
    tensor operator/(const tensor &other) const;
    tensor operator+=(const tensor &other) const;
    tensor operator-() const;
    float &operator[](const size_t idx) const;

    friend tensor operator+(const float sca, const tensor &t);
    friend tensor operator-(const float sca, const tensor &t);
    friend tensor operator*(const float sca, const tensor &t);
    friend tensor operator/(const float sca, const tensor &t);
    friend tensor operator+(const tensor &t, const float sca);
    friend tensor operator/(const tensor &t, const float sca);
    friend std::ostream &operator<<(std::ostream &os, const tensor &t);
};