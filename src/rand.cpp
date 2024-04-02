#include "rand.h"
#include "ten.h"

#include <cassert>
#include <numeric>
#include <random>

Ten NormalDistribution(const std::vector<size_t> &shape, const float mean, const float stdDev)
{
    Ten newTensor = Ten();

    for (const size_t &i : shape)
        assert(i != 0);

    newTensor.shape = shape;

    if (newTensor.shape.size() > 0)
        newTensor.size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    else
        newTensor.size = 1;

    newTensor.elem = new float[newTensor.size];

    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> dist(mean, stdDev);

    for (size_t i = 0; i < newTensor.size; ++i)
        newTensor[i] = dist(rng);

    return newTensor;
}

Ten Shuffle(const Ten &tensor, const size_t randomState)
{
    Ten newTensor = tensor;
    std::mt19937 rng(randomState);

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