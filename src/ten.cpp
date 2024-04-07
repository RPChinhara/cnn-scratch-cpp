#include "ten.h"

#include <cassert>
#include <numeric>
#include <string>

ten::ten(const std::vector<float> elem, const std::vector<size_t> shape)
{
    assert(elem.size() != 0);

    for (const size_t &i : shape)
        assert(i != 0);
    this->shape = std::move(shape);

    if (this->shape.size() > 0)
        size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
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
}

ten::~ten()
{
    if (elem != nullptr)
        delete[] elem;
}

ten::ten(const ten &other)
{
    elem = new float[other.size];
    std::copy(other.elem, other.elem + other.size, elem);
    size = other.size;
    shape = other.shape;
}

ten::ten(ten &&other)
{
    elem = other.elem;
    size = other.size;
    shape = other.shape;

    other.elem = nullptr;
    other.size = 0;
}

ten &ten::operator=(const ten &other)
{
    if (this != &other)
    {
        delete[] elem;
        elem = new float[other.size];
        std::copy(other.elem, other.elem + other.size, elem);
        size = other.size;
        shape = other.shape;
    }
    return *this;
}

ten &ten::operator=(ten &&other)
{
    if (this != &other)
    {
        delete[] elem;

        elem = other.elem;
        size = other.size;
        shape = other.shape;

        other.elem = nullptr;
        other.size = 0;
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

ten ten::operator+(const ten &t) const // it's working
{
    ten newTensor = *this;
    if (ShapeEqual(shape, t.shape))
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] + t[i];
    }
    else
    {
        assert(shape.back() == t.shape.back());
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] + t[i % t.shape.back()];
    }
    return newTensor;
}

// ten ten::operator+(const ten& t) const
// {
//     ten newTensor;
//     if (ShapeEqual(shape, t.shape)) {
//         newTensor = *this;
//         std::cout << "1" << std::endl;
//         for (size_t i = 0; i < newTensor.size; ++i)
//             newTensor[i] = elem[i] + t[i];
//     } else {
//         // std::cout << "2" << std::endl;

//         assert(shape.back() == t.shape.back());

//         float *A, *B, *C;
//         cudaMalloc((void**)&A, this->size * sizeof(float));
//         cudaMalloc((void**)&B, this->size * sizeof(float));
//         cudaMalloc((void**)&C, this->size * sizeof(float));
//         cudaMemcpy(A, elem, this->size * sizeof(float), cudaMemcpyHostToDevice);
//         cudaMemcpy(B, t.elem, t.size * sizeof(float), cudaMemcpyHostToDevice);

//         constexpr int blockSize = 128;
//         int gridSize = (this->size + blockSize - 1) / blockSize;
//         OperatorPlus<<<gridSize, blockSize>>>(A, B, C, t.shape.back(), this->size);

//         cudaError_t cudaError = cudaGetLastError();
//         if (cudaError != cudaSuccess)
//           td::cerr << "CUDA kernel launch error " + std::string(cudaGetErrorString(cudaError)) <<
//                       std::endl;

//         newTensor = *this;
//         cudaMemcpy(newTensor.elem, C, newTensor.size * sizeof(float), cudaMemcpyDeviceToHost);
//         cudaFree(A);
//         cudaFree(B);
//         cudaFree(C);

//     }

//     return newTensor;
// }

ten ten::operator-(const ten &t) const
{
    ten newTensor = *this;
    if (ShapeEqual(shape, t.shape)) // it's working
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] - t[i];
    }
    else if (shape.back() == t.shape.back()) // it's working
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] - t[i % t.shape.back()];
    }
    else if (shape.front() == t.shape.front()) // it's working
    {
        for (size_t i = 0; i < newTensor.size; ++i)
        {
            size_t idx = i / shape.back();
            newTensor[i] = elem[i] - t[idx];
        }
    }
    return newTensor;
}

ten ten::operator*(const ten &t) const
{
    ten newTensor = *this;
    if (ShapeEqual(shape, t.shape))
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] * t[i];
    }
    else
    {
        assert(shape.back() == t.shape.back());
        size_t idx = 0;
        for (size_t i = 0; i < newTensor.size; ++i)
        {
            if (idx == t.shape.back())
                idx = 0;
            newTensor[i] = elem[i] * t[idx];
            ++idx;
        }
    }
    return newTensor;
}

ten ten::operator/(const ten &t) const
{
    ten newTensor = *this;
    if (ShapeEqual(shape, t.shape))
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] / t[i];
    }
    else
    {
        size_t idx = 0;
        if (shape.back() == t.shape.back())
        {
            for (size_t i = 0; i < newTensor.size; ++i)
            {
                if (idx == t.shape.back())
                    idx = 0;
                newTensor[i] = elem[i] / t[idx];
                ++idx;
            }
        }
        else if (shape.front() == t.shape.front())
        {
            for (size_t i = 0; i < newTensor.size; ++i)
            {
                if (i % shape.back() == 0 && i != 0)
                    ++idx;
                newTensor[i] = elem[i] / t[idx];
            }
        }
        else
        {
            std::cerr << "Shapes don't much." << std::endl;
        }
    }
    return newTensor;
}

ten ten::operator+=(const ten &other) const
{
    for (size_t i = 0; i < size; ++i)
        elem[i] = elem[i] + other[i];
    return *this;
}

ten ten::operator-=(const ten &other) const
{
    assert(ShapeEqual(shape, other.shape));
    for (size_t i = 0; i < size; ++i)
        elem[i] = elem[i] - other[i];
    return *this;
}

float &ten::operator[](const size_t idx) const
{
    return elem[idx];
}

ten operator-(const float sca, const ten &t)
{
    ten newTensor = t;
    for (size_t i = 0; i < t.size; ++i)
        newTensor[i] = sca - t[i];
    return newTensor;
}

ten operator*(const float sca, const ten &t)
{
    ten newTensor = t;
    for (size_t i = 0; i < t.size; ++i)
        newTensor[i] = sca * t[i];
    return newTensor;
}

void operator/(const ten &t, const float sca)
{
    for (size_t i = 0; i < t.size; ++i)
        t[i] = t[i] / sca;
    std::cout << "fjdkfjkdjf" << std::endl;
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

std::ostream &operator<<(std::ostream &os, const ten &t)
{
    size_t idx = 0;
    if (t.shape.size() == 0)
    {
        os << "Tensor(" << std::to_string(t[0]) << ", shape=())";
        return os;
    }
    else
    {
        if (t.size == 1)
        {
            os << "Tensor(";
            for (size_t i = 0; i < t.shape.size(); ++i)
                os << "[";
        }
        else
        {
            os << "Tensor(\n";
            for (size_t i = 0; i < t.shape.size(); ++i)
                os << "[";
        }

        if (t.size == 1)
        {
            for (size_t i = 0; i < t.size; ++i)
                if (i == t.size - 1)
                    os << std::to_string(t[i]);
                else
                    os << std::to_string(t[i]) << " ";
        }
        else
        {
            std::vector<size_t> num_elem_each_batch = GetNumElemEachBatch(t.shape);
            size_t num_elem_most_inner_mat = GetNumElemMostInnerMat(t.shape);

            for (size_t i = 0; i < t.size; ++i)
            {
                bool num_elem_each_batch_done = false;
                size_t num_square_brackets = 0;

                if (t.shape.size() > 2)
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

                if (i % t.shape.back() == 0 && i != 0 && !(i % num_elem_most_inner_mat == 0))
                {
                    os << "]\n";

                    for (size_t i = 0; i < t.shape.size() - 1; ++i)
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
                        for (size_t i = 0; i < t.shape.size() - num_square_brackets - 1; ++i)
                            os << " ";
                        for (size_t i = 0; i < num_square_brackets + 1; ++i)
                            os << "[";
                    }
                    else
                    {
                        os << "\n";
                        for (size_t i = 0; i < t.shape.size() - 2; ++i)
                            os << " ";
                        os << "[[";
                    }
                }

                if (i == t.size - 1)
                {
                    os << std::to_string(t[i]);
                    continue;
                }

                if (idx == t.shape.back())
                    idx = 0;

                if (t.shape.back() == 1)
                {
                    os << std::to_string(t[i]);
                }
                else
                {
                    if (idx % (t.shape.back() - 1) == 0 && idx != 0)
                        os << std::to_string(t[i]);
                    else
                        os << std::to_string(t[i]) << " ";
                }
                ++idx;

                num_elem_each_batch_done = false;
            }
        }

        for (size_t i = 0; i < t.shape.size(); ++i)
            os << "]";
    }

    os << ", shape=(";
    for (size_t i = 0; i < t.shape.size(); ++i)
    {
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