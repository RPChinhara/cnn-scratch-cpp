#include "ten.h"

#include <cassert>
#include <numeric>
#include <string>

Ten::Ten(const std::vector<float> elem, const std::vector<size_t> shape)
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

Ten::~Ten()
{
    if (elem != nullptr)
        delete[] elem;
}

Ten::Ten(const Ten &other)
{
    elem = new float[other.size];
    std::copy(other.elem, other.elem + other.size, elem);
    size = other.size;
    shape = other.shape;
}

Ten::Ten(Ten &&other)
{
    elem = other.elem;
    size = other.size;
    shape = other.shape;

    other.elem = nullptr;
    other.size = 0;
}

Ten &Ten::operator=(const Ten &other)
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

Ten &Ten::operator=(Ten &&other)
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

Ten Ten::operator+(const Ten &ten) const
{
    Ten newTensor = *this;
    if (ShapeEqual(shape, ten.shape))
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] + ten[i];
    }
    else
    {
        assert(shape.back() == ten.shape.back());
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] + ten[i % ten.shape.back()];
    }
    return newTensor;
}

// Ten Ten::operator+(const Ten& ten) const
// {
//     Ten newTensor;
//     if (ShapeEqual(shape, ten.shape)) {
//         newTensor = *this;
//         std::cout << "1" << std::endl;
//         for (size_t i = 0; i < newTensor.size; ++i)
//             newTensor[i] = elem[i] + ten[i];
//     } else {
//         // std::cout << "2" << std::endl;

//         assert(shape.back() == ten.shape.back());

//         float *A, *B, *C;
//         cudaMalloc((void**)&A, this->size * sizeof(float));
//         cudaMalloc((void**)&B, this->size * sizeof(float));
//         cudaMalloc((void**)&C, this->size * sizeof(float));
//         cudaMemcpy(A, elem, this->size * sizeof(float), cudaMemcpyHostToDevice);
//         cudaMemcpy(B, ten.elem, ten.size * sizeof(float), cudaMemcpyHostToDevice);

//         constexpr int blockSize = 128;
//         int gridSize = (this->size + blockSize - 1) / blockSize;
//         OperatorPlus<<<gridSize, blockSize>>>(A, B, C, ten.shape.back(), this->size);

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

Ten Ten::operator-(const Ten &ten) const
{
    Ten newTensor = *this;
    if (ShapeEqual(shape, ten.shape))
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] - ten[i];
    }
    else if (shape.back() == ten.shape.back())
    {
        size_t idx = 0;
        for (size_t i = 0; i < newTensor.size; ++i)
        {
            if (idx == ten.shape.back())
                idx = 0;
            newTensor[i] = elem[i] - ten[idx];
            ++idx;
        }
    }
    else if (shape.front() == ten.shape.front())
    {
        size_t idx = 0;
        for (size_t i = 0; i < shape.front(); ++i)
        {
            for (size_t j = 0; j < shape.back(); ++j)
            {
                newTensor[idx] = elem[idx] - ten[i];
                ++idx;
            }
        }
    }
    return newTensor;
}

Ten Ten::operator*(const Ten &ten) const
{
    Ten newTensor = *this;
    if (ShapeEqual(shape, ten.shape))
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] * ten[i];
    }
    else
    {
        assert(shape.back() == ten.shape.back());
        size_t idx = 0;
        for (size_t i = 0; i < newTensor.size; ++i)
        {
            if (idx == ten.shape.back())
                idx = 0;
            newTensor[i] = elem[i] * ten[idx];
            ++idx;
        }
    }
    return newTensor;
}

Ten Ten::operator/(const Ten &ten) const
{
    Ten newTensor = *this;
    if (ShapeEqual(shape, ten.shape))
    {
        for (size_t i = 0; i < newTensor.size; ++i)
            newTensor[i] = elem[i] / ten[i];
    }
    else
    {
        size_t idx = 0;
        if (shape.back() == ten.shape.back())
        {
            for (size_t i = 0; i < newTensor.size; ++i)
            {
                if (idx == ten.shape.back())
                    idx = 0;
                newTensor[i] = elem[i] / ten[idx];
                ++idx;
            }
        }
        else if (shape.front() == ten.shape.front())
        {
            for (size_t i = 0; i < newTensor.size; ++i)
            {
                if (i == shape.back())
                    ++idx;
                newTensor[i] = elem[i] / ten[idx];
            }
        }
        else
        {
            std::cerr << "Shapes don't much." << std::endl;
        }
    }
    return newTensor;
}

Ten Ten::operator+=(const Ten &other) const
{
    for (size_t i = 0; i < size; ++i)
        elem[i] = elem[i] + other[i];
    return *this;
}

Ten Ten::operator-=(const Ten &other) const
{
    assert(ShapeEqual(shape, other.shape));
    for (size_t i = 0; i < size; ++i)
        elem[i] = elem[i] - other[i];
    return *this;
}

float &Ten::operator[](const size_t idx) const
{
    return elem[idx];
}

Ten operator-(const float sca, const Ten &ten)
{
    Ten newTensor = ten;
    for (size_t i = 0; i < ten.size; ++i)
        newTensor[i] = sca - ten[i];
    return newTensor;
}

Ten operator*(const float sca, const Ten &ten)
{
    Ten newTensor = ten;
    for (size_t i = 0; i < ten.size; ++i)
        newTensor[i] = sca * ten[i];
    return newTensor;
}

void operator/(const Ten &ten, const float sca)
{
    for (size_t i = 0; i < ten.size; ++i)
        ten[i] = ten[i] / sca;
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

std::ostream &operator<<(std::ostream &os, const Ten &ten)
{
    size_t idx = 0;
    if (ten.shape.size() == 0)
    {
        os << "Tensor(" << std::to_string(ten[0]) << ", shape=())";
        return os;
    }
    else
    {
        if (ten.size == 1)
        {
            os << "Tensor(";
            for (size_t i = 0; i < ten.shape.size(); ++i)
                os << "[";
        }
        else
        {
            os << "Tensor(\n";
            for (size_t i = 0; i < ten.shape.size(); ++i)
                os << "[";
        }

        if (ten.size == 1)
        {
            for (size_t i = 0; i < ten.size; ++i)
                if (i == ten.size - 1)
                    os << std::to_string(ten[i]);
                else
                    os << std::to_string(ten[i]) << " ";
        }
        else
        {
            std::vector<size_t> num_elem_each_batch = GetNumElemEachBatch(ten.shape);
            size_t num_elem_most_inner_mat = GetNumElemMostInnerMat(ten.shape);

            for (size_t i = 0; i < ten.size; ++i)
            {
                bool num_elem_each_batch_done = false;
                size_t num_square_brackets = 0;

                if (ten.shape.size() > 2)
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

                if (i % ten.shape.back() == 0 && i != 0 && !(i % num_elem_most_inner_mat == 0))
                {
                    os << "]\n";

                    for (size_t i = 0; i < ten.shape.size() - 1; ++i)
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
                        for (size_t i = 0; i < ten.shape.size() - num_square_brackets - 1; ++i)
                            os << " ";
                        for (size_t i = 0; i < num_square_brackets + 1; ++i)
                            os << "[";
                    }
                    else
                    {
                        os << "\n";
                        for (size_t i = 0; i < ten.shape.size() - 2; ++i)
                            os << " ";
                        os << "[[";
                    }
                }

                if (i == ten.size - 1)
                {
                    os << std::to_string(ten[i]);
                    continue;
                }

                if (idx == ten.shape.back())
                    idx = 0;

                if (ten.shape.back() == 1)
                {
                    os << std::to_string(ten[i]);
                }
                else
                {
                    if (idx % (ten.shape.back() - 1) == 0 && idx != 0)
                        os << std::to_string(ten[i]);
                    else
                        os << std::to_string(ten[i]) << " ";
                }
                ++idx;

                num_elem_each_batch_done = false;
            }
        }

        for (size_t i = 0; i < ten.shape.size(); ++i)
            os << "]";
    }

    os << ", shape=(";
    for (size_t i = 0; i < ten.shape.size(); ++i)
    {
        if (i != ten.shape.size() - 1)
            os << ten.shape[i] << ", ";
        else if (ten.shape.size() == 1)
            os << ten.shape[i] << ",";
        else
            os << ten.shape[i];
    }
    os << "))";

    return os;
}