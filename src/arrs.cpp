#include "arrs.h"
#include "ten.h"

#include <cassert>
#include <numeric>

ten clip_by_value(const ten &t, float clip_val_min, float clip_val_max)
{
    assert(clip_val_min <= clip_val_max);
    ten t_new = t;

    for (auto i = 0; i < t.size; ++i)
    {
        if (t[i] < clip_val_min)
            t_new[i] = clip_val_min;
        else if (t[i] > clip_val_max)
            t_new[i] = clip_val_max;
    }

    return t_new;
}

ten slice(const ten &t, const size_t begin, const size_t size)
{
    ten newTensor = zeros({size, t.shape.back()});

    for (auto i = begin * t.shape.back(); i < (begin * t.shape.back()) + (size * t.shape.back()); ++i)
        newTensor[i - (begin * t.shape.back())] = t[i];

    return newTensor;
}

ten zeros(const std::vector<size_t> &shape)
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
    std::fill(newTensor.elem, newTensor.elem + newTensor.size, 0.0f);

    return newTensor;
}