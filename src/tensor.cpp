#include "tensor.h"
#include "arrs.h"
#include "math.h"

#include <iomanip>
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

static bool ShapeEqual(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) {
    bool equal = false;
    if (std::equal(shape1.begin(), shape1.end(), shape2.begin()))
        equal = true;
    return equal;
}

tensor tensor::operator+(const tensor& other) const {
    tensor t_new = *this;

    if (ShapeEqual(shape, other.shape)) {
        // for (auto i = 0; i < size; ++i)
        //     t_new[i] = elems[i] + other[i];
        t_new = add(*this, other);
    } else if (shape.front() == other.shape.front()) {
        // std::cout << 22222222222 << "\n";
        for (auto i = 0; i < size; ++i) {
            size_t idx = i / shape.back();
            t_new[i] = elems[i] + other[idx];
        }
    } else if (shape.back() == other.shape.back()) {
        // std::cout << 33333333 << "\n";
        for (auto i = 0; i < size; ++i)
            t_new[i] = elems[i] + other[i % other.shape.back()];
    }

    // NOTE: if given (2, 2) and (2, 1), make t_new with shape muches with bigger one in this case (2, 2)

    return t_new;
}

tensor tensor::operator-(const tensor& other) const {
    tensor t_new = *this;

    if (ShapeEqual(shape, other.shape)) {
        // for (auto i = 0; i < size; ++i)
        //     t_new[i] = elems[i] - other[i];
        t_new = subtract(*this, other);
    } else if (shape.front() == other.shape.front()) {
        for (auto i = 0; i < size; ++i) {
            size_t idx = i / shape.back();
            t_new[i] = elems[i] - other[idx];
        }
    } else if (shape.back() == other.shape.back()) {
        for (auto i = 0; i < size; ++i)
            t_new[i] = elems[i] - other[i % other.shape.back()];
    }

    return t_new;
}

tensor tensor::operator*(const tensor& other) const {
    tensor t_new = *this;

    if (ShapeEqual(shape, other.shape)) {
        // for (auto i = 0; i < size; ++i)
        //     t_new[i] = elems[i] * other[i];
        t_new = multiply(*this, other);
    } else if (shape.back() == other.shape.back()) {
        for (auto i = 0; i < size; ++i)
            t_new[i] = elems[i] * other[i % other.shape.back()];
    }

    return t_new;
}

tensor tensor::operator/(const tensor& other) const {
    tensor t_new = *this;

    if (ShapeEqual(shape, other.shape)) {
        // for (auto i = 0; i < size; ++i)
        //     t_new[i] = elems[i] / other[i];
        t_new = divide(*this, other);
    } else if (shape.front() == other.shape.front()) {
        for (auto i = 0; i < size; ++i) {
            size_t idx = i / shape.back();
            t_new[i] = elems[i] / other[idx];
        }
    } else if (shape.back() == other.shape.back()) {
        for (auto i = 0; i < size; ++i)
            t_new[i] = elems[i] / other[i % other.shape.back()];
    }

    return t_new;
}

tensor& tensor::operator+=(const tensor& other) {
    // for (auto i = 0; i < size; ++i)
    //     elems[i] += other[i];
    // return *this;

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

float& tensor::operator()(const size_t i, const size_t j) {
    return elems[i * shape.back() + j];
}

const float& tensor::operator()(const size_t i, const size_t j) const {
    return elems[i * shape.back() + j];
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
    os << std::fixed << std::setprecision(8);

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