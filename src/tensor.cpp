#include "tensor.h"

#include <cassert>
#include <string>

Tensor::Tensor(const std::vector<float> elem, const std::vector<size_t> shape)
{
    assert(elem.size() != 0);
    
    for (const size_t& i : shape)
        assert(i != 0);
    this->shape = std::move(shape);

    if (this->shape.size() > 0) {
        size_t num_elem = 1;
        for (const size_t& i : shape)
            num_elem *= i;
        size = num_elem;
    } else
        size = 1;

    if (elem.size() == 1) {
        this->elem = new float[size];
        std::fill(this->elem, this->elem + size, *elem.data());
    } else {
        assert(size == elem.size());
        this->elem = new float[size];
        memcpy(this->elem, elem.data(), size * sizeof(float));
    } 

    if (this->shape.size() > 0) {
        num_ch_dim = 1;
        for (const size_t& i : shape)
            num_ch_dim *= i;
    } else
        num_ch_dim = 0;
}

Tensor::~Tensor()
{
    if (elem != nullptr) 
        delete[] elem;
}

Tensor::Tensor(const Tensor& other)
{
    elem = new float[other.size];
    std::copy(other.elem, other.elem + other.size, elem);
    num_ch_dim = other.num_ch_dim;
    size = other.size;
    shape = other.shape;
}

Tensor::Tensor(Tensor&& other)
{
    elem = other.elem;
    num_ch_dim = other.num_ch_dim;
    size = other.size;
    shape = other.shape;

    other.elem       = nullptr;
    other.num_ch_dim = 0;
    other.size       = 0;
}

Tensor& Tensor::operator=(const Tensor& other)
{
    if (this != &other) {
        delete[] elem;
        elem = new float[other.size];
        std::copy(other.elem, other.elem + other.size, elem);
        num_ch_dim = other.num_ch_dim;
        size = other.size;
        shape = other.shape;
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other)
{
    if (this != &other) {
        delete[] elem;

        elem = other.elem;
        num_ch_dim = other.num_ch_dim;
        size = other.size;
        shape = other.shape;

        other.elem = nullptr;
        other.num_ch_dim = 0;
        other.size = 0;
    }
    return *this;
}

static bool ShapeEqual(const std::vector<size_t>& shape_1, const std::vector<size_t>& shape_2)
{
    bool equal = false;
    if (std::equal(shape_1.begin(), shape_1.end(), shape_2.begin()))
         equal = true;
    return equal;
}

Tensor Tensor::operator+(const Tensor& other) const
{
    Tensor out = *this;
    if (ShapeEqual(shape, other.shape)) {
        for (size_t i = 0; i < out.size; ++i)
            out[i] = elem[i] + other[i];
    } else {
        assert(shape.back() == other.shape.back());
        for (size_t i = 0; i < out.size; ++i)
            out[i] = elem[i] + other[i % other.shape.back()];
    }
    return out;
}

// Tensor Tensor::operator+(const Tensor& other) const
// {
//     Tensor out;
//     if (ShapeEqual(shape, other.shape)) {
//         out = *this;
//         std::cout << "1" << std::endl;
//         for (size_t i = 0; i < out.size; ++i)
//             out[i] = elem[i] + other[i];
//     } else {
//         // std::cout << "2" << std::endl;

//         assert(shape.back() == other.shape.back());

//         float *A, *B, *C;
//         cudaMalloc((void**)&A, this->size * sizeof(float));
//         cudaMalloc((void**)&B, this->size * sizeof(float));
//         cudaMalloc((void**)&C, this->size * sizeof(float));
//         cudaMemcpy(A, elem, this->size * sizeof(float), cudaMemcpyHostToDevice);
//         cudaMemcpy(B, other.elem, other.size * sizeof(float), cudaMemcpyHostToDevice);

//         int blockSize = 128;
//         int gridSize = (this->size + blockSize - 1) / blockSize;
//         OperatorPlus<<<gridSize, blockSize>>>(A, B, C, other.shape.back(), this->size);

//         cudaError_t cudaError = cudaGetLastError();
//         if (cudaError != cudaSuccess)
//             std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(cudaError) << std::endl;

//         out = *this;
//         cudaMemcpy(out.elem, C, out.size * sizeof(float), cudaMemcpyDeviceToHost);
//         cudaFree(A);
//         cudaFree(B);
//         cudaFree(C);

//     }

//     return out;
// }

Tensor Tensor::operator-(const Tensor& other) const
{
    Tensor out = *this;
    if (ShapeEqual(shape, other.shape)) {
        for (size_t i = 0; i < out.size; ++i)
            out[i] = elem[i] - other[i];
    } else if (shape.back() == other.shape.back()) {
        size_t idx = 0;
        for (size_t i = 0; i < out.size; ++i) {
            if (idx == other.shape.back())
                idx = 0;
            out[i] = elem[i] - other[idx];
            ++idx;
        }
    } else if (shape.front() == other.shape.front()) {
        size_t idx = 0;
        for (size_t i = 0; i < shape.front(); ++i) {
            for (size_t j = 0; j < shape.back(); ++j) {
                out[idx] = elem[idx] - other[i];
                ++idx;
            }
        }
    }
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const
{
    Tensor out = *this;
    if (ShapeEqual(shape, other.shape)) {
        for (size_t i = 0; i < out.size; ++i)
            out[i] = elem[i] * other[i];
    } else {
        assert(shape.back() == other.shape.back());
        size_t idx = 0;
        for (size_t i = 0; i < out.size; ++i) {
            if (idx == other.shape.back())
                idx = 0;
            out[i] = elem[i] * other[idx];
            ++idx;
        }
    }
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const
{
    Tensor out = *this;
    if (ShapeEqual(shape, other.shape)) {
        for (size_t i = 0; i < out.size; ++i)
            out[i] = elem[i] / other[i];
    } else {
        size_t idx = 0;
        if (shape.back() == other.shape.back()) {
            for (size_t i = 0; i < out.size; ++i) {
                if (idx == other.shape.back())
                    idx = 0;
                out[i] = elem[i] / other[idx];
                ++idx;
            }
        } else if (shape.front() == other.shape.front()) {
            for (size_t i = 0; i < out.size; ++i) {
                if (i == shape.back())
                    ++idx;
                out[i] = elem[i] / other[idx];
            }
        } else {
            std::cerr << "Shapes don't much." << '\n';
            std::exit(EXIT_FAILURE);
        }
    }
    return out;
}

Tensor Tensor::operator+=(const Tensor& other) const
{
    for (size_t i = 0; i < size; ++i)
        elem[i] = elem[i] + other[i];
    return *this;
}

Tensor Tensor::operator-=(const Tensor& other) const
{
    assert(ShapeEqual(shape, other.shape));
    for (size_t i = 0; i < size; ++i)
        elem[i] = elem[i] - other[i];
    return *this;
}

float& Tensor::operator[](const size_t idx) const
{
    return elem[idx];
}

Tensor operator-(const float sca, const Tensor& other)
{
    Tensor out = other;
    for (size_t i = 0; i < out.size; ++i)
        out[i] = sca - other[i];
    return out;    
}

Tensor operator*(const float sca, const Tensor& other)
{
    Tensor out = other;
    for (size_t i = 0; i < out.size; ++i)
        out[i] = sca * other[i];
    return out;    
}

static size_t GetNumElemMostInnerMat(const std::vector<size_t>& shape)
{
    size_t last_shape = shape[shape.size() - 1];
    size_t second_last_shape = shape[shape.size() - 2];
    return second_last_shape * last_shape;
}

static std::vector<size_t> GetNumElemEachBatch(const std::vector<size_t>& shape)
{
    size_t num_elem = GetNumElemMostInnerMat(shape);
    std::vector<size_t> num_elem_each_batch;

    for (auto it = std::rbegin(shape) + 2; it != std::rend(shape); ++it) {
        num_elem *= *it;
        num_elem_each_batch.push_back(num_elem);
    }
    return num_elem_each_batch;
}

std::ostream& operator<<(std::ostream& os, const Tensor& in)
{
    size_t idx = 0;
    if (in.shape.size() == 0) {
        os <<  "Tensor(" << std::to_string(in[0]) << ", shape=())";
        return os;
    } else {
        if (in.num_ch_dim == 1) {
            os << "Tensor(";
            for (size_t i = 0; i < in.shape.size(); ++i)
                os << "[";
        } else {
            os << "Tensor(\n";
            for (size_t i = 0; i < in.shape.size(); ++i)
                os << "[";
        }

        if (in.num_ch_dim == 1) {
            for (size_t i = 0; i < in.size; ++i)
                if (i == in.size - 1)
                    os << std::to_string(in[i]);
                else
                    os << std::to_string(in[i]) << " ";
        } else {
            std::vector<size_t> num_elem_each_batch = GetNumElemEachBatch(in.shape);
            size_t num_elem_most_inner_mat = GetNumElemMostInnerMat(in.shape);

            for (size_t i = 0; i < in.size; ++i) {
                bool num_elem_each_batch_done = false;
                size_t  num_square_brackets = 0;

                if (in.shape.size() > 2) {
                    for (int j = num_elem_each_batch.size() - 1; j >= 0; --j) {
                        if (i % num_elem_each_batch[j] == 0 && i != 0) {
                            num_elem_each_batch_done = true;
                            num_square_brackets = j + 2;
                            break;
                        }
                    }
                }

                if (i % in.shape.back() == 0 && i != 0 && !(i % num_elem_most_inner_mat == 0)) {
                    os << "]\n";

                    for (size_t i = 0; i < in.shape.size() - 1; ++i)
                        os << " ";

                    os << "[";
                } else if (i % num_elem_most_inner_mat == 0 && i != 0) {
                    if (num_elem_each_batch_done) {
                        os << "]";
                        for (size_t i = 0; i < num_square_brackets; ++i)
                            os << "]";

                        os << "\n";
                    } else {
                        os << "]]\n";
                    }
                }

                if (i % num_elem_most_inner_mat == 0 && i != 0) {
                    if (num_elem_each_batch_done) {
                        for (size_t i = 0; i < num_square_brackets; ++i)
                            os << "\n";
                        for (size_t i = 0; i < in.shape.size() - num_square_brackets - 1; ++i)
                            os << " ";
                        for (size_t i = 0; i < num_square_brackets + 1; ++i)
                            os << "[";
                    } else {
                        os << "\n";
                        for (size_t i = 0; i < in.shape.size() - 2; ++i)
                            os << " ";
                        os << "[[";
                    }
                }

                if (i == in.size - 1) {
                    os << std::to_string(in[i]);
                    continue;
                }

                if (idx == in.shape.back()) 
                    idx = 0;

                if (in.shape.back() == 1) {
                    os << std::to_string(in[i]);
                } else {
                    if (idx % (in.shape.back() - 1) == 0 && idx != 0)
                        os << std::to_string(in[i]);
                    else
                        os << std::to_string(in[i]) << " ";
                }
                ++idx;

                num_elem_each_batch_done = false;
            }
        }

        for (size_t i = 0; i < in.shape.size(); ++i)
            os << "]";
    }
    
    os << ", shape=(";
    for (size_t i = 0; i < in.shape.size(); ++i) {
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