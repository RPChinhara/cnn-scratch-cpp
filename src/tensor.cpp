#include "tensor.h"

#include <cassert>
#include <string>
#include <windows.h>

Tensor::Tensor(const std::vector<float> elem, const std::vector<size_t> shape)
{
    assert(elem.size() != 0);

    for (const size_t &i : shape)
        assert(i != 0);
    this->shape = std::move(shape);

    if (this->shape.size() > 0)
    {
        size_t num_elem = 1;
        for (const size_t &i : shape)
            num_elem *= i;
        size = num_elem;
    }
    else
        size = 1;

    if (elem.size() == 1)
    {
        this->elem = new float[size];
        std::fill(this->elem, this->elem + size, *elem.data());
    }
    else
    {
        assert(size == elem.size());
        this->elem = new float[size];
        memcpy(this->elem, elem.data(), size * sizeof(float));
    }

    if (this->shape.size() > 0)
    {
        num_ch_dim = 1;
        for (const size_t &i : shape)
            num_ch_dim *= i;
    }
    else
        num_ch_dim = 0;
}

Tensor::~Tensor()
{
    if (elem != nullptr)
        delete[] elem;
}

Tensor::Tensor(const Tensor &tensor)
{
    elem = new float[tensor.size];
    std::copy(tensor.elem, tensor.elem + tensor.size, elem);
    num_ch_dim = tensor.num_ch_dim;
    size = tensor.size;
    shape = tensor.shape;
}

Tensor::Tensor(Tensor &&tensor)
{
    elem = tensor.elem;
    num_ch_dim = tensor.num_ch_dim;
    size = tensor.size;
    shape = tensor.shape;

    tensor.elem = nullptr;
    tensor.num_ch_dim = 0;
    tensor.size = 0;
}

Tensor &Tensor::operator=(const Tensor &tensor)
{
    if (this != &tensor)
    {
        delete[] elem;
        elem = new float[tensor.size];
        std::copy(tensor.elem, tensor.elem + tensor.size, elem);
        num_ch_dim = tensor.num_ch_dim;
        size = tensor.size;
        shape = tensor.shape;
    }
    return *this;
}

Tensor &Tensor::operator=(Tensor &&tensor)
{
    if (this != &tensor)
    {
        delete[] elem;

        elem = tensor.elem;
        num_ch_dim = tensor.num_ch_dim;
        size = tensor.size;
        shape = tensor.shape;

        tensor.elem = nullptr;
        tensor.num_ch_dim = 0;
        tensor.size = 0;
    }
    return *this;
}

static bool ShapeEqual(const std::vector<size_t> &shape1, const std::vector<size_t> &shape2)
{
    bool equal = false;
    if (std::equal(shape1.begin(), shape1.end(), shape2.begin()))
        equal = true;
    return equal;
}

Tensor Tensor::operator+(const Tensor &tensor) const
{
    Tensor newTensor = *this;
    if (ShapeEqual(shape, tensor.shape))
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] + tensor[i];
    }
    else
    {
        assert(shape.back() == tensor.shape.back());
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] + tensor[i % tensor.shape.back()];
    }
    return newTensor;
}

// Tensor Tensor::operator+(const Tensor& tensor) const
// {
//     Tensor newTensor;
//     if (ShapeEqual(shape, tensor.shape)) {
//         newTensor = *this;
//         std::cout << "1" << std::endl;
//         for (size_t i = 0; i < newTensor.size; ++i)
//             newTensor[i] = elem[i] + tensor[i];
//     } else {
//         // std::cout << "2" << std::endl;

//         assert(shape.back() == tensor.shape.back());

//         float *A, *B, *C;
//         cudaMalloc((void**)&A, this->size * sizeof(float));
//         cudaMalloc((void**)&B, this->size * sizeof(float));
//         cudaMalloc((void**)&C, this->size * sizeof(float));
//         cudaMemcpy(A, elem, this->size * sizeof(float), cudaMemcpyHostToDevice);
//         cudaMemcpy(B, tensor.elem, tensor.size * sizeof(float), cudaMemcpyHostToDevice);

//         constexpr int blockSize = 128;
//         int gridSize = (this->size + blockSize - 1) / blockSize;
//         OperatorPlus<<<gridSize, blockSize>>>(A, B, C, tensor.shape.back(), this->size);

//         cudaError_t cudaError = cudaGetLastError();
//         if (cudaError != cudaSuccess)
//             MessageBox(nullptr, ("CUDA kernel launch error " + std::string(cudaGetErrorString(cudaError))).c_str(),
//                      "Error", MB_ICONERROR);

//         newTensor = *this;
//         cudaMemcpy(newTensor.elem, C, newTensor.size * sizeof(float), cudaMemcpyDeviceToHost);
//         cudaFree(A);
//         cudaFree(B);
//         cudaFree(C);

//     }

//     return newTensor;
// }

Tensor Tensor::operator-(const Tensor &tensor) const
{
    Tensor newTensor = *this;
    if (ShapeEqual(shape, tensor.shape))
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] - tensor[i];
    }
    else if (shape.back() == tensor.shape.back())
    {
        size_t idx = 0;
        for (size_t i = 0; i < newTensor.size; ++i)
        {
            if (idx == tensor.shape.back())
                idx = 0;
            newTensor[i] = elem[i] - tensor[idx];
            ++idx;
        }
    }
    else if (shape.front() == tensor.shape.front())
    {
        size_t idx = 0;
        for (size_t i = 0; i < shape.front(); ++i)
        {
            for (size_t j = 0; j < shape.back(); ++j)
            {
                newTensor[idx] = elem[idx] - tensor[i];
                ++idx;
            }
        }
    }
    return newTensor;
}

Tensor Tensor::operator*(const Tensor &tensor) const
{
    Tensor newTensor = *this;
    if (ShapeEqual(shape, tensor.shape))
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] * tensor[i];
    }
    else
    {
        assert(shape.back() == tensor.shape.back());
        size_t idx = 0;
        for (size_t i = 0; i < newTensor.size; ++i)
        {
            if (idx == tensor.shape.back())
                idx = 0;
            newTensor[i] = elem[i] * tensor[idx];
            ++idx;
        }
    }
    return newTensor;
}

Tensor Tensor::operator/(const Tensor &tensor) const
{
    Tensor newTensor = *this;
    if (ShapeEqual(shape, tensor.shape))
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] / tensor[i];
    }
    else
    {
        size_t idx = 0;
        if (shape.back() == tensor.shape.back())
        {
            for (size_t i = 0; i < newTensor.size; ++i)
            {
                if (idx == tensor.shape.back())
                    idx = 0;
                newTensor[i] = elem[i] / tensor[idx];
                ++idx;
            }
        }
        else if (shape.front() == tensor.shape.front())
        {
            for (size_t i = 0; i < newTensor.size; ++i)
            {
                if (i == shape.back())
                    ++idx;
                newTensor[i] = elem[i] / tensor[idx];
            }
        }
        else
        {
            MessageBox(nullptr, "Shapes don't much", "Error", MB_ICONERROR);
        }
    }
    return newTensor;
}

Tensor Tensor::operator+=(const Tensor &tensor) const
{
    for (size_t i = 0; i < size; ++i)
        elem[i] = elem[i] + tensor[i];
    return *this;
}

Tensor Tensor::operator-=(const Tensor &tensor) const
{
    assert(ShapeEqual(shape, tensor.shape));
    for (size_t i = 0; i < size; ++i)
        elem[i] = elem[i] - tensor[i];
    return *this;
}

float &Tensor::operator[](const size_t idx) const
{
    return elem[idx];
}

Tensor operator-(const float sca, const Tensor &tensor)
{
    Tensor newTensor = tensor;
    for (size_t i = 0; i < tensor.size; ++i)
        newTensor[i] = sca - tensor[i];
    return newTensor;
}

Tensor operator*(const float sca, const Tensor &tensor)
{
    Tensor newTensor = tensor;
    for (size_t i = 0; i < tensor.size; ++i)
        newTensor[i] = sca * tensor[i];
    return newTensor;
}

void operator/(const Tensor &tensor, const float sca)
{
    for (size_t i = 0; i < tensor.size; ++i)
        tensor[i] = tensor[i] / sca;
}

static size_t GetNumElemMostInnerMat(const std::vector<size_t> &shape)
{
    size_t last_shape = shape[shape.size() - 1];
    size_t second_last_shape = shape[shape.size() - 2];
    return second_last_shape * last_shape;
}

static std::vector<size_t> GetNumElemEachBatch(const std::vector<size_t> &shape)
{
    size_t num_elem = GetNumElemMostInnerMat(shape);
    std::vector<size_t> num_elem_each_batch;

    for (auto it = std::rbegin(shape) + 2; it != std::rend(shape); ++it)
    {
        num_elem *= *it;
        num_elem_each_batch.push_back(num_elem);
    }
    return num_elem_each_batch;
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor)
{
    size_t idx = 0;
    if (tensor.shape.size() == 0)
    {
        os << "Tensor(" << std::to_string(tensor[0]) << ", shape=())";
        return os;
    }
    else
    {
        if (tensor.num_ch_dim == 1)
        {
            os << "Tensor(";
            for (size_t i = 0; i < tensor.shape.size(); ++i)
                os << "[";
        }
        else
        {
            os << "Tensor(\n";
            for (size_t i = 0; i < tensor.shape.size(); ++i)
                os << "[";
        }

        if (tensor.num_ch_dim == 1)
        {
            for (size_t i = 0; i < tensor.size; ++i)
                if (i == tensor.size - 1)
                    os << std::to_string(tensor[i]);
                else
                    os << std::to_string(tensor[i]) << " ";
        }
        else
        {
            std::vector<size_t> num_elem_each_batch = GetNumElemEachBatch(tensor.shape);
            size_t num_elem_most_inner_mat = GetNumElemMostInnerMat(tensor.shape);

            for (size_t i = 0; i < tensor.size; ++i)
            {
                bool num_elem_each_batch_done = false;
                size_t num_square_brackets = 0;

                if (tensor.shape.size() > 2)
                {
                    for (int j = num_elem_each_batch.size() - 1; j >= 0; --j)
                    {
                        if (i % num_elem_each_batch[j] == 0 && i != 0)
                        {
                            num_elem_each_batch_done = true;
                            num_square_brackets = j + 2;
                            break;
                        }
                    }
                }

                if (i % tensor.shape.back() == 0 && i != 0 && !(i % num_elem_most_inner_mat == 0))
                {
                    os << "]\n";

                    for (size_t i = 0; i < tensor.shape.size() - 1; ++i)
                        os << " ";

                    os << "[";
                }
                else if (i % num_elem_most_inner_mat == 0 && i != 0)
                {
                    if (num_elem_each_batch_done)
                    {
                        os << "]";
                        for (size_t i = 0; i < num_square_brackets; ++i)
                            os << "]";

                        os << "\n";
                    }
                    else
                    {
                        os << "]]\n";
                    }
                }

                if (i % num_elem_most_inner_mat == 0 && i != 0)
                {
                    if (num_elem_each_batch_done)
                    {
                        for (size_t i = 0; i < num_square_brackets; ++i)
                            os << "\n";
                        for (size_t i = 0; i < tensor.shape.size() - num_square_brackets - 1; ++i)
                            os << " ";
                        for (size_t i = 0; i < num_square_brackets + 1; ++i)
                            os << "[";
                    }
                    else
                    {
                        os << "\n";
                        for (size_t i = 0; i < tensor.shape.size() - 2; ++i)
                            os << " ";
                        os << "[[";
                    }
                }

                if (i == tensor.size - 1)
                {
                    os << std::to_string(tensor[i]);
                    continue;
                }

                if (idx == tensor.shape.back())
                    idx = 0;

                if (tensor.shape.back() == 1)
                {
                    os << std::to_string(tensor[i]);
                }
                else
                {
                    if (idx % (tensor.shape.back() - 1) == 0 && idx != 0)
                        os << std::to_string(tensor[i]);
                    else
                        os << std::to_string(tensor[i]) << " ";
                }
                ++idx;

                num_elem_each_batch_done = false;
            }
        }

        for (size_t i = 0; i < tensor.shape.size(); ++i)
            os << "]";
    }

    os << ", shape=(";
    for (size_t i = 0; i < tensor.shape.size(); ++i)
    {
        if (i != tensor.shape.size() - 1)
            os << tensor.shape[i] << ", ";
        else if (tensor.shape.size() == 1)
            os << tensor.shape[i] << ",";
        else
            os << tensor.shape[i];
    }
    os << "))";

    return os;
}