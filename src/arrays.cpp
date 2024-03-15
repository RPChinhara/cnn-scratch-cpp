#include "arrays.h"
#include "tensor.h"

#include <cassert>
#include <numeric>

Tensor ClipByValue(const Tensor &tensor, float clipValMin, float clipValMax)
{
    assert(clipValMin <= clipValMax);
    Tensor newTensor = tensor;

    for (size_t i = 0; i < tensor.size; ++i)
    {
        if (tensor[i] < clipValMin)
            newTensor[i] = clipValMin;
        else if (tensor[i] > clipValMax)
            newTensor[i] = clipValMax;
    }

    return newTensor;
}

Tensor Slice(const Tensor &tensor, const size_t begin, const size_t size)
{
    Tensor newTensor = Zeros({size, tensor.shape.back()});

    for (size_t i = begin * tensor.shape.back(); i < (begin * tensor.shape.back()) + (size * tensor.shape.back()); ++i)
        newTensor[i - (begin * tensor.shape.back())] = tensor[i];

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

Tensor Transpose(const Tensor &tensor)
{
    assert(tensor.shape.size() >= 2);

    Tensor newTensor = Zeros({tensor.shape.back(), tensor.shape[tensor.shape.size() - 2]});

    std::vector<size_t> idx_rows;

    for (size_t i = 0; i < tensor.size; ++i)
        idx_rows.push_back(i * tensor.shape.back());

    size_t batchSize = GetBatchSize(tensor.shape);

    size_t idx = 0;

    for (size_t i = 0; i < batchSize; ++i)
    {
        for (size_t j = 0; j < newTensor.shape[newTensor.shape.size() - 2]; ++j)
        {
            for (size_t k = 0; k < newTensor.shape.back(); ++k)
            {
                newTensor[idx] = tensor[idx_rows[k + (i * newTensor.shape.back())]];
                idx_rows[k + (i * newTensor.shape.back())] += 1;
                ++idx;
            }
        }
    }

    return newTensor;
}

Tensor Zeros(const std::vector<size_t> &shape)
{
    Tensor newTensor = Tensor();

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