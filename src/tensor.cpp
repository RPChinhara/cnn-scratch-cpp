#include "tensor.h"
#include "linalg.h"

#include <cassert>

Tensor::Tensor(const std::vector<float> elem, const std::vector<unsigned int> shape) {
    assert(elem.size() != 0);
    
    _shape.reserve(shape.size());
    for (unsigned int elem : shape)
        assert(elem != 0);
    _shape = std::move(shape);

    if (_shape.size() > 0) {
        unsigned int num_elem = 1;
        for (unsigned int elem : shape)
            num_elem *= elem;
        _size = num_elem;
    } else
        _size = 1;

    if (elem.size() == 1) {
        _elem = new float[_size];
        std::fill(_elem, _elem + _size, *elem.data());
    } else {
        assert(_size == elem.size());
        _elem = new float[_size];
        memcpy(_elem, elem.data(), sizeof(float) * _size);
    } 

    if (_shape.size() > 0) {
        _num_ch_dim = 1;
        for (int i = 0; i < shape.size() - 1; ++i)
            _num_ch_dim *= shape[i];
    } else
        _num_ch_dim = 0;
}

Tensor::Tensor(const Tensor& o) {
    float *ptr = new float[o._size];
    memcpy(ptr, o._elem, sizeof(float) * o._size);
    _elem       = ptr;
    _num_ch_dim = o._num_ch_dim;
    _size       = o._size;
    _shape      = o._shape;
}

Tensor::Tensor(Tensor&& o) noexcept :
    _elem(o._elem),
    _num_ch_dim(o._num_ch_dim),
    _size(o._size),
    _shape(std::move(o._shape)) {
        o._elem       = nullptr;
        o._num_ch_dim = 0;
        o._size       = 0;
    }

Tensor::~Tensor() {
    if (_elem != nullptr) 
        delete[] _elem;
}

static bool shape_eq(const std::vector<unsigned int>& shape1, const std::vector<unsigned int>& shape2) {
    bool eq{};
    if (std::equal(shape1.begin(), shape1.end(), shape2.begin()))
         eq = true;
    return eq;
}

Tensor Tensor::operator+(const Tensor& o) const {
    Tensor out = *this;
    if (shape_eq(_shape, o._shape)) {
        for (unsigned int i = 0; i < out._size; ++i)
            out[i] = _elem[i] + o[i];
    } else {
        assert(_shape.back() == o._shape.back());
        for (unsigned int i = 0; i < out._size; ++i)
            out[i] = _elem[i] + o[i % o._shape.back()];
    }
    return out;
}

Tensor Tensor::operator-(const Tensor& o) const {
    Tensor out = *this;
    if (shape_eq(_shape, o._shape)) {
        for (unsigned int i = 0; i < out._size; ++i)
            out[i] = _elem[i] - o[i];
    } else if (_shape.back() == o._shape.back()) {
        unsigned short idx = 0;
        for (unsigned int i = 0; i < out._size; ++i) {
            if (idx == o._shape.back())
                idx = 0;
            out[i] = _elem[i] - o[idx];
            ++idx;
        }
    } else if (_shape.front() == o._shape.front()) {
        unsigned short idx = 0;
        for (unsigned int i = 0; i < _shape.front(); ++i) {
            for (unsigned int j = 0; j < _shape.back(); ++j) {
                out[idx] = _elem[idx] - o[i];
                ++idx;
            }
        }
    }
    return out;
}

Tensor Tensor::operator*(const Tensor& o) const {
    Tensor out = *this;
    if (shape_eq(_shape, o._shape)) {
        for (unsigned int i = 0; i < out._size; ++i)
            out[i] = _elem[i] * o[i];
    } else {
        assert(_shape.back() == o._shape.back());
        unsigned short idx = 0;
        for (unsigned int i = 0; i < out._size; ++i) {
            if (idx == o._shape.back())
                idx = 0;
            out[i] = _elem[i] * o[idx];
            ++idx;
        }
    }
    return out;
}

Tensor Tensor::operator/(const Tensor& o) const {
    Tensor out = *this;
    if (shape_eq(_shape, o._shape)) {
        for (unsigned int i = 0; i < out._size; ++i)
            out[i] = _elem[i] / o[i];
    } else {
        unsigned short idx = 0;
        if (_shape.back() == o._shape.back()) {
            for (unsigned int i = 0; i < out._size; ++i) {
                if (idx == o._shape.back())
                    idx = 0;
                out[i] = _elem[i] / o[idx];
                ++idx;
            }
        } else if (_shape.front() == o._shape.front()) {
            for (unsigned int i = 0; i < out._size; ++i) {
                if (i == _shape.back())
                    ++idx;
                out[i] = _elem[i] / o[idx];
            }
        } else {
            std::cerr << "Shapes don't much." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    return out;
}

Tensor& Tensor::operator=(const Tensor& o) {
    float *ptr = new float[o._size];
    memcpy(ptr, o._elem, sizeof(float) * o._size);
    _elem       = ptr;
    _num_ch_dim = o._num_ch_dim;
    _size       = o._size;
    _shape      = o._shape;
    return *this;
}

Tensor Tensor::operator+=(const Tensor& o) const {
    for (unsigned int i = 0; i < _size; ++i)
        _elem[i] = _elem[i] + o[i];
    return *this;
}

Tensor Tensor::operator-=(const Tensor& o) const {
    assert(shape_eq(_shape, o._shape));
    for (unsigned int i = 0; i < _size; ++i)
        _elem[i] = _elem[i] - o[i];
    return *this;
}

float& Tensor::operator[](const unsigned int ind) const {
    return _elem[ind];
}

Tensor operator-(const float sca, const Tensor& o) {
    Tensor out = o;
    for (unsigned int i = 0; i < out._size; ++i)
        out[i] = sca - o[i];
    return out;    
}

Tensor operator*(const float sca, const Tensor& o) {
    Tensor out = o;
    for (unsigned int i = 0; i < out._size; ++i)
        out[i] = sca * o[i];
    return out;    
}

static unsigned int get_num_elem_most_inner_mat(const std::vector<unsigned int>& shape) {
    unsigned int last_shape        = shape[shape.size() - 1];
    unsigned int second_last_shape = shape[shape.size() - 2];
    return second_last_shape * last_shape;
}

// Get number of elements for each batch size e.g., if the most inner matrix is [[7, 7, 7], [7, 7, 7]] and shape (2, 2, 2, 2, 3) it'd return 12, 24, and 48.
static std::vector<int> get_num_elem_each_batch(const std::vector<unsigned int>& shape) {
    unsigned int num_elem = get_num_elem_most_inner_mat(shape);
    std::vector<int> num_elem_each_batch;
    // Iterate in reverse order
    for (auto it = std::rbegin(shape) + 2; it != std::rend(shape); ++it) {
        num_elem *= *it;
        num_elem_each_batch.push_back(num_elem);
    }
    return num_elem_each_batch;
}

std::ostream& operator<<(std::ostream& os, const Tensor& in) {
    unsigned short idx{};
    if (in._shape.size() == 0) {
        os <<  "Tensor(" << in[0] << ", shape=())";
        return os;
    } else {
        if (in._num_ch_dim == 1) {
            os << "Tensor(";
            for (unsigned short i = 0; i < in._shape.size(); ++i)
                os << "[";
        } else {
            os << "Tensor(\n";
            for (unsigned short i = 0; i < in._shape.size(); ++i)
                os << "[";
        }

        if (in._num_ch_dim == 1) {
            for (unsigned int i = 0; i < in._size; ++i)
                if (i == in._size - 1)
                    os << in[i];
                else
                    os << in[i] << " ";
        } else {
            std::vector<int> num_elem_each_batch = get_num_elem_each_batch(in._shape);
            unsigned int num_elem_most_inner_mat = get_num_elem_most_inner_mat(in._shape);


            for (unsigned int i = 0; i < in._size; ++i) {
                bool num_elem_each_batch_done{};
                unsigned short  num_square_brackets{};

                if (in._shape.size() > 2) {
                    for (short j = num_elem_each_batch.size() - 1; j >= 0; --j) {
                        if (i % num_elem_each_batch[j] == 0 && i != 0) {
                            num_elem_each_batch_done = true;
                            // This will be the number of ']' needs for each batches e.g., shape=(2, 2, 2, 2, 3), then
                            // if it's divisible by 12 add "]]"   0 + 2 where 0 is 'j'.
                            // if it's divisible by 24 add "]]]"  1 + 2 where 1 is 'j'.
                            // if it's divisible by 48 add "]]]]" 2 + 2 where 2 is 'j'.
                            num_square_brackets = j + 2;
                            break;
                        }
                    }
                }

                // Make new lines and add ']' by each cases.
                if (i % in._shape.back() == 0 && i != 0 && !(i % num_elem_most_inner_mat == 0)) {
                    // Make new lines and add ']' for each vectors.
                    os << "]\n";
                    for (unsigned short i = 0; i < in._shape.size() - 1; ++i)
                        os << " ";
                    os << "[";
                } else if (i % num_elem_most_inner_mat == 0 && i != 0) {
                    // Make new lines and add ']' for each matrices.
                    if (num_elem_each_batch_done) {
                        // For each vectors.
                        os << "]";
                        for (unsigned short i = 0; i < num_square_brackets; ++i)
                            os << "]";
                        os << "\n";
                    } else 
                        os << "]]\n";
                }

                // Make new lines, add spaces, and add '[' for every matrix.
                if (i % num_elem_most_inner_mat == 0 && i != 0) {
                    if (num_elem_each_batch_done) {
                        for (unsigned short i = 0; i < num_square_brackets; ++i)
                            os << "\n";
                        for (unsigned short i = 0; i < in._shape.size() - num_square_brackets - 1; ++i)
                            os << " ";
                        for (unsigned short i = 0; i < num_square_brackets + 1; ++i)
                            os << "[";
                    } else {
                        os << "\n";
                        for (unsigned short i = 0; i < in._shape.size() - 2; ++i)
                            os << " ";
                        os << "[[";
                    }
                }

                // If 'i' is last, then print without a space.
                if (i == in._size - 1) {
                    os << in[i];
                    continue;
                }

                // Print elements.
                if (idx == in._shape.back()) 
                    idx = 0;

                if (in._shape.back() == 1)
                    os << in[i];
                else {
                    // Print w/o spaces if it's last element of the row.
                    if (idx % (in._shape.back() - 1) == 0 && idx != 0)
                        os << in[i];
                    else
                        os << in[i] << " ";
                }
                ++idx;

                num_elem_each_batch_done = false;
            }
        }

        // Add ']' after the last element.
        for (unsigned short i = 0; i < in._shape.size(); ++i)
            os << "]";
    }
    
    // Add shape to 'os'.
    os << ", shape=(";
    for (unsigned short i = 0; i < in._shape.size(); ++i) {
        if (i != in._shape.size() - 1)
            os << in._shape[i] << ", ";
        else if (in._shape.size() == 1)
            os << in._shape[i] << ",";
        else
            os << in._shape[i];
    }
    os << "))";

    return os;
}

Tensor Tensor::T() const {
    return Transpose(*this);
}