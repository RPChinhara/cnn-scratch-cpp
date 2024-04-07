#pragma once

#include <iostream>
#include <vector>

class ten
{
  public:
    size_t size;
    std::vector<size_t> shape;
    float *elem = nullptr;

    ten() = default;
    ten(const std::vector<float> elem, const std::vector<size_t> shape);
    ~ten();
    ten(const ten &other);
    ten(ten &&other);
    ten &operator=(const ten &other);
    ten &operator=(ten &&other);
    ten operator+(const ten &other) const;
    ten operator-(const ten &other) const;
    ten operator*(const ten &other) const;
    ten operator/(const ten &other) const;
    ten operator+=(const ten &other) const;
    ten operator-=(const ten &other) const;
    float &operator[](const size_t idx) const;
    friend ten operator-(const float sca, const ten &t);
    friend ten operator*(const float sca, const ten &t);
    friend std::ostream &operator<<(std::ostream &os, const ten &t);
};