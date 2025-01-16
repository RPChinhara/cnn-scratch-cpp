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

    for (auto i = 0; i < shape.size(); ++i) {
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
    for (auto i = 0; i < size; ++i)
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

static size_t get_num_elem_most_inner_mat(const std::vector<size_t>& shape) {
    size_t last_shape = shape[shape.size() - 1];
    size_t second_last_shape = shape[shape.size() - 2];
    return second_last_shape * last_shape;
}

static std::vector<size_t> get_num_elem_each_batch(const std::vector<size_t>& shape) {
    size_t num_elem = get_num_elem_most_inner_mat(shape);
    std::vector<size_t> num_elem_each_batch;

    for (auto it = std::rbegin(shape) + 2; it != std::rend(shape); ++it) {
        num_elem *= *it;
        num_elem_each_batch.push_back(num_elem);
    }
    return num_elem_each_batch;
}

std::ostream& operator<<(std::ostream& os, const tensor& t) {
    size_t idx = 0;

    if (t.shape.size() == 0) {
        os << "Tensor(" << std::to_string(t[0]) << ", shape=())";
        return os;
    } else {
        if (t.size == 1) {
            os << "Tensor(";
            for (auto i = 0; i < t.shape.size(); ++i)
                os << "[";
        } else {
            os << "Tensor(\n";
            for (auto i = 0; i < t.shape.size(); ++i)
                os << "[";
        }

        if (t.size == 1) {
            for (auto i = 0; i < t.size; ++i) {
                if (i == t.size - 1)
                    os << t[i];
                else
                    os << t[i] << " ";
            }
        } else {
            std::vector<size_t> num_elem_each_batch = get_num_elem_each_batch(t.shape);
            size_t num_elem_most_inner_mat = get_num_elem_most_inner_mat(t.shape);

            for (auto i = 0; i < t.size; ++i) {
                bool num_elem_each_batch_done = false;
                size_t num_square_brackets = 0;

                if (2 < t.shape.size()) {
                    for (auto j = num_elem_each_batch.size() - 1; 0 < j; --j) {
                        if (i % num_elem_each_batch[j] == 0 && i != 0) {
                            num_elem_each_batch_done = true;
                            num_square_brackets = j + 2;
                            break;
                        }
                    }
                }

                if (i % t.shape.back() == 0 && i != 0 && !(i % num_elem_most_inner_mat == 0)) {
                    os << "]\n";

                    for (auto i = 0; i < t.shape.size() - 1; ++i)
                        os << " ";

                    os << "[";
                } else if (i % num_elem_most_inner_mat == 0 && i != 0) {
                    if (num_elem_each_batch_done) {
                        os << "]";
                        for (auto i = 0; i < num_square_brackets; ++i)
                            os << "]";

                        os << "\n";
                    } else {
                        os << "]]\n";
                    }
                }

                if (i % num_elem_most_inner_mat == 0 && i != 0) {
                    if (num_elem_each_batch_done) {
                        for (auto i = 0; i < num_square_brackets; ++i)
                            os << "\n";
                        for (auto i = 0; i < t.shape.size() - num_square_brackets - 1; ++i)
                            os << " ";
                        for (auto i = 0; i < num_square_brackets + 1; ++i)
                            os << "[";
                    } else {
                        os << "\n";
                        for (auto i = 0; i < t.shape.size() - 2; ++i)
                            os << " ";
                        os << "[[";
                    }
                }

                if (i == t.size - 1) {
                    os << t[i];
                    continue;
                }

                if (idx == t.shape.back())
                    idx = 0;

                if (t.shape.back() == 1) {
                    os << t[i];
                } else {
                    if (idx % (t.shape.back() - 1) == 0 && idx != 0)
                        os << t[i];
                    else
                        os << t[i] << " ";
                }

                ++idx;
                num_elem_each_batch_done = false;
            }
        }

        for (auto i = 0; i < t.shape.size(); ++i)
            os << "]";
    }

    os << ", shape=(";

    for (auto i = 0; i < t.shape.size(); ++i) {
        if (i != t.shape.size() - 1)
            os << t.shape[i] << ", ";
        else if (t.shape.size() == 1)
            os << t.shape[i] << ",";
        else
            os << t.shape[i];
    }

    os << "))";

    return os;
}