#include "rd.h"
#include "tensor.h"

#include <cassert>
#include <numeric>
#include <random>

tensor normal_dist(const std::vector<size_t> &shape, const float mean, const float std_dev)
{
    tensor t_new = tensor();

    for (auto i : shape)
        assert(i != 0);

    t_new.shape = shape;

    if (0 < t_new.shape.size())
        t_new.size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    else
        t_new.size = 1;

    t_new.elem = new float[t_new.size];

    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<float> dist(mean, std_dev);

    for (auto i = 0; i < t_new.size; ++i)
        t_new[i] = dist(rng);

    return t_new;
}

tensor uniform_dist(const std::vector<size_t> &shape, const float min_val, const float max_val)
{
    tensor t_new = tensor();

    for (auto i : shape)
        assert(i != 0);

    t_new.shape = shape;

    if (0 < t_new.shape.size())
        t_new.size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    else
        t_new.size = 1;

    t_new.elem = new float[t_new.size];

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> dist(min_val, max_val);

    for (auto i = 0; i < t_new.size; ++i)
        t_new[i] = dist(rng);

    return t_new;
}

tensor shuffle(const tensor &t, const size_t rd_state)
{
    tensor t_new = t;
    std::mt19937 rng(rd_state);

    for (auto i = t.shape.front() - 1; 0 < i; --i)
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