#include "arrs.h"
#include "ten.h"

#include <cassert>
#include <numeric>

Ten ClipByValue(const Ten &ten, float clipValMin, float clipValMax)
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

static size_t GetBatchSize(const std::vector<size_t> &shape)
{
    assert(shape.size() > 1);
    size_t batchSize = 1;

    for (size_t i = 0; i < shape.size() - 2; ++i)
        batchSize *= shape[i];

    return batchSize;
}

Ten Transpose(const Ten &ten)
{
    assert(ten.shape.size() >= 2);

    Ten newTensor = Zeros({ten.shape.back(), ten.shape[ten.shape.size() - 2]});

    std::vector<size_t> idx_rows;

    for (size_t i = 0; i < ten.size; ++i)
        idx_rows.push_back(i * ten.shape.back());

    size_t batchSize = GetBatchSize(ten.shape);

    size_t idx = 0;

    for (size_t i = 0; i < batchSize; ++i)
    {
        for (size_t j = 0; j < newTensor.shape[newTensor.shape.size() - 2]; ++j)
        {
            for (size_t k = 0; k < newTensor.shape.back(); ++k)
            {
                newTensor[idx] = ten[idx_rows[k + (i * newTensor.shape.back())]];
                idx_rows[k + (i * newTensor.shape.back())] += 1;
                ++idx;
            }
        }
    }

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