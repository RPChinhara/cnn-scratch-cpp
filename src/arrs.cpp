#include "arrs.h"
#include "ten.h"

#include <cassert>
#include <numeric>

Ten clip_by_value(const Ten &ten, float clipValMin, float clipValMax)
{
    assert(clipValMin <= clipValMax);
    Ten newTensor = ten;

    for (size_t i = 0; i < ten.size; ++i)
    {
        if (ten[i] < clipValMin)
            newTensor[i] = clipValMin;
        else if (ten[i] > clipValMax)
            newTensor[i] = clipValMax;
    }

    return newTensor;
}

Ten slice(const Ten &ten, const size_t begin, const size_t size)
{
    Ten newTensor = Zeros({size, ten.shape.back()});

    for (size_t i = begin * ten.shape.back(); i < (begin * ten.shape.back()) + (size * ten.shape.back()); ++i)
        newTensor[i - (begin * ten.shape.back())] = ten[i];

    return newTensor;
}

Ten Zeros(const std::vector<size_t> &shape)
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