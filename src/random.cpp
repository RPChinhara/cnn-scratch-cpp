#include "random.h"
#include "tensor.h"

#include <cassert>
#include <random>

Tensor NormalDistribution(const std::vector<size_t> &shape, const float mean, const float stddev)
{
    Tensor out = Tensor();

    for (const size_t &i : shape)
        assert(i != 0);

    out.shape = shape;

    if (out.shape.size() > 0)
    {
        size_t num_elem = 1;

        for (const size_t &i : shape)
            num_elem *= i;

        out.size = num_elem;
    }
    else
    {
        out.size = 1;
    }

    out.elem = new float[out.size];

    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> dist(mean, stddev);

    for (size_t i = 0; i < out.size; ++i)
        out[i] = dist(rng);

    if (out.shape.size() > 0)
    {
        out.num_ch_dim = 1;

        for (const size_t &i : shape)
            out.num_ch_dim *= i;
    }
    else
    {
        out.num_ch_dim = 0;
    }

    return out;
}

Tensor Shuffle(const Tensor &in, const size_t random_state)
{
    Tensor out = in;
    std::mt19937 rng(random_state);

    for (size_t i = in.shape.front() - 1; i > 0; --i)
    {
        std::uniform_int_distribution<> dist(0, i);
        int j = dist(rng);

        for (size_t k = 0; k < in.shape.back(); ++k)
        {
            float temp = out[(in.shape.back() - 1) * i + i + k];
            out[(in.shape.back() - 1) * i + i + k] = out[(in.shape.back() - 1) * j + j + k];
            out[(in.shape.back() - 1) * j + j + k] = temp;
        }
    }
    return out;
}