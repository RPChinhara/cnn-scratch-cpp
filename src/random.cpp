#include "random.h"
#include "tensor.h"

#include <cassert>
#include <random>

Tensor NormalDistribution(const std::vector<size_t> &shape, const float mean, const float stddev)
{
    Tensor newTensor = Tensor();

    for (const size_t &i : shape)
        assert(i != 0);

    newTensor.shape = shape;

    if (newTensor.shape.size() > 0)
    {
        size_t num_elem = 1;

        for (const size_t &i : shape)
            num_elem *= i;

        newTensor.size = num_elem;
    }
    else
    {
        newTensor.size = 1;
    }

    newTensor.elem = new float[newTensor.size];

    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> dist(mean, stddev);

    for (size_t i = 0; i < newTensor.size; ++i)
        newTensor[i] = dist(rng);

    if (newTensor.shape.size() > 0)
    {
        newTensor.num_ch_dim = 1;

        for (const size_t &i : shape)
            newTensor.num_ch_dim *= i;
    }
    else
    {
        newTensor.num_ch_dim = 0;
    }

    return newTensor;
}

Tensor Shuffle(const Tensor &tensor, const size_t random_state)
{
    Tensor newTensor = tensor;
    std::mt19937 rng(random_state);

    for (size_t i = tensor.shape.front() - 1; i > 0; --i)
    {
        std::uniform_int_distribution<> dist(0, i);
        int j = dist(rng);

        for (size_t k = 0; k < tensor.shape.back(); ++k)
        {
            float temp = newTensor[(tensor.shape.back() - 1) * i + i + k];
            newTensor[(tensor.shape.back() - 1) * i + i + k] = newTensor[(tensor.shape.back() - 1) * j + j + k];
            newTensor[(tensor.shape.back() - 1) * j + j + k] = temp;
        }
    }
    return newTensor;
}