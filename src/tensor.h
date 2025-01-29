#pragma once

#include <iostream>
#include <vector>

class tensor {
  public:
    size_t size;
    std::vector<size_t> shape;
    float* elems = nullptr;

    tensor() = default;
    ~tensor();
    tensor(const tensor& other);
    tensor(tensor&& other) noexcept;
    tensor& operator=(const tensor& other);
    tensor& operator=(tensor&& other) noexcept;

    const std::string get_shape() const;
    tensor& reshape(const std::vector<size_t>& new_shape);

    tensor operator+(const tensor& other) const;
    tensor operator-(const tensor& other) const;
    tensor operator*(const tensor& other) const;
    tensor operator/(const tensor& other) const;
    tensor& operator+=(const tensor& other);
    tensor operator-() const;
    float& operator[](const size_t idx) const;
    float& operator()(const size_t i, const size_t j);
    const float& operator()(const size_t i, const size_t j) const;

    float get(const std::vector<size_t>& indices) const;
    void set(const std::vector<size_t>& indices, float value) const;

    tensor slice(size_t row_start, size_t row_end, size_t col_start, size_t col_end) const;
    tensor slice_rows(size_t row_start, size_t row_end) const;
    tensor slice_cols(size_t col_start, size_t col_end) const;

    friend tensor operator+(const float sca, const tensor& t);
    friend tensor operator-(const float sca, const tensor& t);
    friend tensor operator*(const float sca, const tensor& t);
    friend tensor operator/(const float sca, const tensor& t);
    friend tensor operator+(const tensor& t, const float sca);
    friend tensor operator/(const tensor& t, const float sca);
    friend std::ostream& operator<<(std::ostream& os, const tensor& t);

  private:
    size_t calculate_flat_index(const std::vector<size_t>& indices) const;
};