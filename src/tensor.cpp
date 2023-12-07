#include "tensor.h"
#include "linalg.h"

#include <cassert>

Tensor::Tensor(const std::vector<float> elem, const std::vector<unsigned int> shape)
{
    assert(elem.size() != 0);
    
    this->shape.reserve(shape.size());
    for (unsigned int elem : shape)
        assert(elem != 0);
    this->shape = std::move(shape);

    if (this->shape.size() > 0) {
        unsigned int num_elem = 1;
        for (unsigned int elem : shape)
            num_elem *= elem;
        size = num_elem;
    } else
        size = 1;

    if (elem.size() == 1) {
        this->elem = new float[size];
        std::fill(this->elem, this->elem + size, *elem.data());
    } else {
        assert(size == elem.size());
        this->elem = new float[size];
        memcpy(this->elem, elem.data(), sizeof(float) * size);
    } 

    if (this->shape.size() > 0) {
        num_ch_dim = 1;
        for (int i = 0; i < shape.size() - 1; ++i)
            num_ch_dim *= shape[i];
    } else
        num_ch_dim = 0;
}

Tensor::Tensor(const Tensor& o)
{
    float *ptr = new float[o.size];
    memcpy(ptr, o.elem, sizeof(float) * o.size);
    elem = ptr;
    num_ch_dim = o.num_ch_dim;
    size = o.size;
    shape = o.shape;
}

Tensor::Tensor(Tensor&& o) noexcept :
    elem(o.elem),
    num_ch_dim(o.num_ch_dim),
    size(o.size),
    shape(std::move(o.shape)) {
        o.elem       = nullptr;
        o.num_ch_dim = 0;
        o.size       = 0;
    }

Tensor::~Tensor()
{
    if (elem != nullptr) 
        delete[] elem;
}

static bool ShapeEqual(const std::vector<unsigned int>& shape1, const std::vector<unsigned int>& shape2)
{
    bool eq{};
    if (std::equal(shape1.begin(), shape1.end(), shape2.begin()))
         eq = true;
    return eq;
}

Tensor Tensor::operator+(const Tensor& o) const
{
    Tensor out = *this;
    if (ShapeEqual(shape, o.shape)) {
        for (unsigned int i = 0; i < out.size; ++i)
            out[i] = elem[i] + o[i];
    } else {
        assert(shape.back() == o.shape.back());
        for (unsigned int i = 0; i < out.size; ++i)
            out[i] = elem[i] + o[i % o.shape.back()];
    }
    return out;
}

Tensor Tensor::operator-(const Tensor& o) const
{
    Tensor out = *this;
    if (ShapeEqual(shape, o.shape)) {
        for (unsigned int i = 0; i < out.size; ++i)
            out[i] = elem[i] - o[i];
    } else if (shape.back() == o.shape.back()) {
        unsigned short idx = 0;
        for (unsigned int i = 0; i < out.size; ++i) {
            if (idx == o.shape.back())
                idx = 0;
            out[i] = elem[i] - o[idx];
            ++idx;
        }
    } else if (shape.front() == o.shape.front()) {
        unsigned short idx = 0;
        for (unsigned int i = 0; i < shape.front(); ++i) {
            for (unsigned int j = 0; j < shape.back(); ++j) {
                out[idx] = elem[idx] - o[i];
                ++idx;
            }
        }
    }
    return out;
}

Tensor Tensor::operator*(const Tensor& o) const
{
    Tensor out = *this;
    if (ShapeEqual(shape, o.shape)) {
        for (unsigned int i = 0; i < out.size; ++i)
            out[i] = elem[i] * o[i];
    } else {
        assert(shape.back() == o.shape.back());
        unsigned short idx = 0;
        for (unsigned int i = 0; i < out.size; ++i) {
            if (idx == o.shape.back())
                idx = 0;
            out[i] = elem[i] * o[idx];
            ++idx;
        }
    }
    return out;
}

Tensor Tensor::operator/(const Tensor& o) const
{
    Tensor out = *this;
    if (ShapeEqual(shape, o.shape)) {
        for (unsigned int i = 0; i < out.size; ++i)
            out[i] = elem[i] / o[i];
    } else {
        unsigned short idx = 0;
        if (shape.back() == o.shape.back()) {
            for (unsigned int i = 0; i < out.size; ++i) {
                if (idx == o.shape.back())
                    idx = 0;
                out[i] = elem[i] / o[idx];
                ++idx;
            }
        } else if (shape.front() == o.shape.front()) {
            for (unsigned int i = 0; i < out.size; ++i) {
                if (i == shape.back())
                    ++idx;
                out[i] = elem[i] / o[idx];
            }
        } else {
            std::cerr << "Shapes don't much." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
    return out;
}

Tensor& Tensor::operator=(const Tensor& o)
{
    float *ptr = new float[o.size];
    memcpy(ptr, o.elem, sizeof(float) * o.size);
    elem       = ptr;
    num_ch_dim = o.num_ch_dim;
    size       = o.size;
    shape      = o.shape;
    return *this;
}

Tensor Tensor::operator+=(const Tensor& o) const
{
    for (unsigned int i = 0; i < size; ++i)
        elem[i] = elem[i] + o[i];
    return *this;
}

Tensor Tensor::operator-=(const Tensor& o) const
{
    assert(ShapeEqual(shape, o.shape));
    for (unsigned int i = 0; i < size; ++i)
        elem[i] = elem[i] - o[i];
    return *this;
}

float& Tensor::operator[](const unsigned int idx) const
{
    return elem[idx];
}

Tensor operator-(const float sca, const Tensor& o)
{
    Tensor out = o;
    for (unsigned int i = 0; i < out.size; ++i)
        out[i] = sca - o[i];
    return out;    
}

Tensor operator*(const float sca, const Tensor& o)
{
    Tensor out = o;
    for (unsigned int i = 0; i < out.size; ++i)
        out[i] = sca * o[i];
    return out;    
}

static unsigned int GetNumElemMostInnerMat(const std::vector<unsigned int>& shape)
{
    unsigned int last_shape = shape[shape.size() - 1];
    unsigned int second_last_shape = shape[shape.size() - 2];
    return second_last_shape * last_shape;
}

static std::vector<int> GetNumElemEachBatch(const std::vector<unsigned int>& shape)
{
    unsigned int num_elem = GetNumElemMostInnerMat(shape);
    std::vector<int> num_elem_each_batch;

    for (auto it = std::rbegin(shape) + 2; it != std::rend(shape); ++it) {
        num_elem *= *it;
        num_elem_each_batch.push_back(num_elem);
    }
    return num_elem_each_batch;
}

std::ostream& operator<<(std::ostream& os, const Tensor& in)
{
    unsigned short idx{};
    if (in.shape.size() == 0) {
        os <<  "Tensor(" << in[0] << ", shape=())";
        return os;
    } else {
        if (in.num_ch_dim == 1) {
            os << "Tensor(";
            for (unsigned short i = 0; i < in.shape.size(); ++i)
                os << "[";
        } else {
            os << "Tensor(\n";
            for (unsigned short i = 0; i < in.shape.size(); ++i)
                os << "[";
        }

        if (in.num_ch_dim == 1) {
            for (unsigned int i = 0; i < in.size; ++i)
                if (i == in.size - 1)
                    os << in[i];
                else
                    os << in[i] << " ";
        } else {
            std::vector<int> num_elem_each_batch = GetNumElemEachBatch(in.shape);
            unsigned int num_elem_most_inner_mat = GetNumElemMostInnerMat(in.shape);

            for (unsigned int i = 0; i < in.size; ++i) {
                bool num_elem_each_batch_done{};
                unsigned short  num_square_brackets{};

                if (in.shape.size() > 2) {
                    for (short j = num_elem_each_batch.size() - 1; j >= 0; --j) {
                        if (i % num_elem_each_batch[j] == 0 && i != 0) {
                            num_elem_each_batch_done = true;
                            num_square_brackets = j + 2;
                            break;
                        }
                    }
                }

                if (i % in.shape.back() == 0 && i != 0 && !(i % num_elem_most_inner_mat == 0)) {
                    os << "]\n";

                    for (unsigned short i = 0; i < in.shape.size() - 1; ++i)
                        os << " ";

                    os << "[";
                } else if (i % num_elem_most_inner_mat == 0 && i != 0) {
                    if (num_elem_each_batch_done) {
                        os << "]";
                        for (unsigned short i = 0; i < num_square_brackets; ++i)
                            os << "]";

                        os << "\n";
                    } else {
                        os << "]]\n";
                    }
                }

                if (i % num_elem_most_inner_mat == 0 && i != 0) {
                    if (num_elem_each_batch_done) {
                        for (unsigned short i = 0; i < num_square_brackets; ++i)
                            os << "\n";
                        for (unsigned short i = 0; i < in.shape.size() - num_square_brackets - 1; ++i)
                            os << " ";
                        for (unsigned short i = 0; i < num_square_brackets + 1; ++i)
                            os << "[";
                    } else {
                        os << "\n";
                        for (unsigned short i = 0; i < in.shape.size() - 2; ++i)
                            os << " ";
                        os << "[[";
                    }
                }

                if (i == in.size - 1) {
                    os << in[i];
                    continue;
                }

                if (idx == in.shape.back()) 
                    idx = 0;

                if (in.shape.back() == 1) {
                    os << in[i];
                } else {
                    if (idx % (in.shape.back() - 1) == 0 && idx != 0)
                        os << in[i];
                    else
                        os << in[i] << " ";
                }
                ++idx;

                num_elem_each_batch_done = false;
            }
        }

        for (unsigned short i = 0; i < in.shape.size(); ++i)
            os << "]";
    }
    
    os << ", shape=(";
    for (unsigned short i = 0; i < in.shape.size(); ++i) {
        if (i != in.shape.size() - 1)
            os << in.shape[i] << ", ";
        else if (in.shape.size() == 1)
            os << in.shape[i] << ",";
        else
            os << in.shape[i];
    }
    os << "))";

    return os;
}