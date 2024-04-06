#include "arrs.h"
#include "ten.h"

#include <cassert>
#include <numeric>

Ten clip_by_value(const Ten &t, float clipValMin, float clipValMax)
{
    assert(clipValMin <= clipValMax);
    Ten newTensor = t;

    for (size_t i = 0; i < t.size; ++i)
    {
        if (t[i] < clipValMin)
            newTensor[i] = clipValMin;
        else if (t[i] > clipValMax)
            newTensor[i] = clipValMax;
    }

    return newTensor;
}

Ten slice(const Ten &t, const size_t begin, const size_t size)
{
    Ten newTensor = zeros({size, t.shape.back()});

    for (size_t i = begin * t.shape.back(); i < (begin * t.shape.back()) + (size * t.shape.back()); ++i)
        newTensor[i - (begin * t.shape.back())] = t[i];

    return newTensor;
}

Ten zeros(const std::vector<size_t> &shape)
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
    std::fill(newTensor.elem, newTensor.elem + newTensor.size, 0.0f);

    return newTensor;
}