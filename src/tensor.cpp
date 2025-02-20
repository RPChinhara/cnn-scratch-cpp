#include "tensor.h"
#include "arrs.h"
#include "math.h"

#include <iomanip>
#include <numeric>
#include <string>

tensor::~tensor() {
    delete[] elems;
    elems = nullptr;
}

tensor::tensor(const tensor& other) {
    elems = new float[other.size];
    std::copy(other.elems, other.elems + other.size, elems);
    size = other.size;
    shape = other.shape;
}

tensor::tensor(tensor&& other) noexcept {
    elems = other.elems;
    size = other.size;
    shape = other.shape;

    other.elems = nullptr;
    other.size = 0;
}

tensor& tensor::operator=(const tensor& other) {
    if (this != &other) {
        delete[] elems;
        elems = new float[other.size];
        std::copy(other.elems, other.elems + other.size, elems);
        size = other.size;
        shape = other.shape;
    }
    return *this;
}

tensor& tensor::operator=(tensor&& other) noexcept {
    if (this != &other) {
        delete[] elems;

        elems = other.elems;
        size = other.size;
        shape = other.shape;

        other.elems = nullptr;
        other.size = 0;
    }
    return *this;
}

const std::string tensor::get_shape() const {
    std::string shapes = "(";

    for (size_t i = 0; i < shape.size(); ++i) {
        if (i == shape.size() - 1)
            shapes += std::to_string(shape[i]);
        else
            shapes += std::to_string(shape[i]) + ", ";
    }

    shapes += ")";

    return shapes;
}

tensor& tensor::reshape(const std::vector<size_t>& new_shape) {
    size_t new_size = 1;
    for (size_t dim : new_shape)
        new_size *= dim;

    shape = new_shape;
    return *this;
}

bool shape_equal(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    bool equal = false;
    if (std::equal(shape1.begin(), shape1.end(), shape2.begin()))
        equal = true;
    return equal;
}

tensor tensor::operator+(const tensor& other) const {
    tensor t_new;

    // (2, 2)    -> 4
    // (2, 1)    -> 3

    // (2, 2)    -> 4
    // (1, 2, 2) -> 5

    if (shape_equal(shape, other.shape)) {
        t_new = add(*this, other);
    } else if (std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(0)) > std::accumulate(other.shape.begin(), other.shape.end(), static_cast<size_t>(0))) {
        tensor other_broadcasted = broadcast_to(other, shape);
        t_new = add(*this, other_broadcasted);
    } else {
        tensor this_broadcasted = broadcast_to(*this, other.shape);
        t_new = add(this_broadcasted, other);
    }

    return t_new;
}

tensor tensor::operator-(const tensor& other) const {
    tensor t_new;

    if (shape_equal(shape, other.shape)) {
        t_new = subtract(*this, other);
    } else if (size > other.size) {
        // NOTE: Intentionally, not changing 'else if' condition like above 'operator+' since this may not be the best solution. For all of these ops, I might just need to compute when the shapes are equal so broadcast before passing to these ops for perm reason and simplicity.
        tensor other_broadcasted = broadcast_to(other, shape);
        t_new = subtract(*this, other_broadcasted);
    } else {
        tensor this_broadcasted = broadcast_to(*this, other.shape);
        t_new = subtract(this_broadcasted, other);
    }

    return t_new;
}

tensor tensor::operator*(const tensor& other) const {
    tensor t_new;

    if (shape_equal(shape, other.shape)) {
        t_new = multiply(*this, other);
    } else if (size > other.size) {
        tensor other_broadcasted = broadcast_to(other, shape);
        t_new = multiply(*this, other_broadcasted);
    } else {
        tensor this_broadcasted = broadcast_to(*this, other.shape);
        t_new = multiply(this_broadcasted, other);
    }

    return t_new;
}

tensor tensor::operator/(const tensor& other) const {
    tensor t_new;

    if (shape_equal(shape, other.shape)) {
        t_new = divide(*this, other);
    } else if (size > other.size) {
        tensor other_broadcasted = broadcast_to(other, shape);
        t_new = divide(*this, other_broadcasted);
    } else {
        tensor this_broadcasted = broadcast_to(*this, other.shape);
        t_new = divide(this_broadcasted, other);
    }

    return t_new;
}

tensor& tensor::operator+=(const tensor& other) {
    *this = add(*this, other);
    return *this;
}

tensor tensor::operator-() const {
    tensor t_new = *this;
    for (size_t i = 0; i < size; ++i)
        t_new[i] = -elems[i];
    return t_new;
}

float& tensor::operator[](const size_t idx) const {
    return elems[idx];
}

// NOTE: Able to overload the () operator to make accessing values more intuitive
// double operator()(const std::vector<int>& indices) const {
//     return get(indices);
// }

// void operator()(const std::vector<int>& indices, double value) {
//     set(indices, value);
// }

// NOTE: Example
// my_tensor({1, 2, 3, 4}) = 42.0;
// double value = my_tensor({1, 2, 3, 4});

float& tensor::operator()(const size_t i, const size_t j) {
    return elems[i * shape.back() + j];
}

const float& tensor::operator()(const size_t i, const size_t j) const {
    return elems[i * shape.back() + j];
}

// Helper method to calculate the flattened index
size_t tensor::calculate_flat_index(const std::vector<size_t>& indices) const {
    size_t flat_index = 0;
    size_t stride = 1;

    // Calculate the flattened index using strides
    for (int i = shape.size() - 1; i >= 0; --i) {
        flat_index += indices[i] * stride;
        stride *= shape[i];
    }

    return flat_index;
}

// NOTE: Just in case used only for 4D tensor
// double get(int n, int c, int h, int w) const {
//     int index = n * shape[1] * shape[2] * shape[3] +
//                 c * shape[2] * shape[3] +
//                 h * shape[3] +
//                 w;
//     return elems[index];
// }

// void set(int n, int c, int h, int w, double value) {
//     int index = n * shape[1] * shape[2] * shape[3] +
//                 c * shape[2] * shape[3] +
//                 h * shape[3] +
//                 w;
//     elems[index] = value;
// }

float tensor::get(const std::vector<size_t>& indices) const {
    size_t index = calculate_flat_index(indices);
    return elems[index];
}

void tensor::set(const std::vector<size_t>& indices, float value) const {
    size_t index = calculate_flat_index(indices);
    elems[index] = value;
}

tensor tensor::slice(size_t row_start, size_t row_end, size_t col_start, size_t col_end) const { // NOTE: Only supports 2D tensor
    size_t new_rows = row_end - row_start;
    size_t new_cols = col_end - col_start;
    tensor result = zeros({new_rows, new_cols});

    for (size_t i = row_start; i < row_end; ++i) {
        for (size_t j = col_start; j < col_end; ++j) {
            result(i - row_start, j - col_start) = this->operator()(i, j);
        }
    }
    return result;
}

tensor tensor::slice_rows(size_t row_start, size_t row_end) const { // NOTE: Only supports 2D tensor
    size_t cols = this->shape.back();
    return slice(row_start, row_end, 0, cols);
}

tensor tensor::slice_cols(size_t col_start, size_t col_end) const { // NOTE: Only supports 2D tensor
    size_t rows = this->shape.front();
    return slice(0, rows, col_start, col_end);
}

tensor operator+(const float sca, const tensor& t) {
    tensor t_sca = fill(t.shape, sca);
    return add(t_sca, t);
}

tensor operator-(const float sca, const tensor& t) {
    tensor t_sca = fill(t.shape, sca);
    return subtract(t_sca, t);
}

tensor operator*(const float sca, const tensor& t) {
    tensor t_sca = fill(t.shape, sca);
    return multiply(t_sca, t);
}

tensor operator/(const float sca, const tensor& t) {
    tensor t_sca = fill(t.shape, sca);
    return divide(t_sca, t);
}

tensor operator+(const tensor& t, const float sca) {
    tensor t_sca = fill(t.shape, sca);
    return add(t, t_sca);
}

tensor operator/(const tensor& t, const float sca) {
    tensor t_sca = fill(t.shape, sca);
    return divide(t, t_sca);
}

std::ostream& operator<<(std::ostream& os, const tensor& t) {
    os << "[";

    size_t mat_size = t.shape.size() < 2 ? 1 : t.shape[t.shape.size() - 2] * t.shape.back();

    for (size_t i = 0; i < t.size; ++i) {
        if (i && i % t.shape.back() == 0) os << "\n ";
        if (i && i % mat_size == 0) os << "\n ";

        if (i == t.size - 1)
            os << std::setw(9) << std::right << t[i];
        else
            os << std::setw(9) << std::right << t[i] << " ";
    }

    os << "] - shape=(";
    for (size_t i = 0; i < t.shape.size(); ++i)
        os << t.shape[i] << (i < t.shape.size() - 1 ? ", " : (t.shape.size() == 1 ? "," : ""));

    os << ")";
    return os;
}