#include "rd.h"
#include "ten.h"

#include <cassert>
#include <numeric>
#include <random>

ten normal_dist(const std::vector<size_t> &shape, const float mean, const float stdDev)
{
    ten newTensor = ten();

    for (auto i : shape)
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

    for (auto i = 0; i < newTensor.size; ++i)
        newTensor[i] = dist(rng);

    return newTensor;
}

ten shuffle(const ten &t, const size_t rd_state)
{
    ten t_new = t;
    std::mt19937 rng(rd_state);

    for (auto i = t.shape.front() - 1; i > 0; --i)
    {
        std::uniform_int_distribution<> dist(0, i);
        int j = dist(rng);

        for (auto k = 0; k < t.shape.back(); ++k)
        {
            float temp = t_new[(t.shape.back() - 1) * i + i + k];
            t_new[(t.shape.back() - 1) * i + i + k] = t_new[(t.shape.back() - 1) * j + j + k];
            t_new[(t.shape.back() - 1) * j + j + k] = temp;
        }
    }
    return t_new;
}