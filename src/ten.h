#pragma once

#include <iostream>
#include <vector>

class Ten
{
  public:
    size_t size;
    std::vector<size_t> shape;
    float *elem = nullptr;

    Ten() = default;
    Ten(const std::vector<float> elem, const std::vector<size_t> shape);
    ~Ten();
    Ten(const Ten &other);
    Ten(Ten &&other);
    Ten &operator=(const Ten &other);
    Ten &operator=(Ten &&other);
    Ten operator+(const Ten &tensor) const;
    Ten operator-(const Ten &tensor) const;
    Ten operator*(const Ten &tensor) const;
    Ten operator/(const Ten &tensor) const;
    Ten operator+=(const Ten &other) const;
    Ten operator-=(const Ten &other) const;
    float &operator[](const size_t idx) const;
    friend Ten operator-(const float sca, const Ten &tensor);
    friend Ten operator*(const float sca, const Ten &tensor);
    friend void operator/(const Ten &tensor, const float sca);
    friend std::ostream &operator<<(std::ostream &os, const Ten &tensor);
};