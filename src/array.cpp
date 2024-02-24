#include "array.h"
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

Tensor Slice(const Tensor &in, const size_t begin, const size_t size)
{
    assert(begin < in.shape[0] && begin + size <= in.shape[0]);
    Tensor out = Zeros({size, in.shape.back()});
    size_t idx = 0;

    for (size_t i = begin * in.shape.back(); i < (begin * in.shape.back()) + (size * in.shape.back()); ++i)
    {
        out[idx] = in[i];
        ++idx;
    }

    return out;
}

static size_t GetBatchSize(const std::vector<size_t> &shape)
{
    assert(shape.size() > 1);
    size_t batch_size = 1;

    for (size_t i = 0; i < shape.size() - 2; ++i)
        batch_size *= shape[i];

    return batch_size;
}

Tensor Transpose(const Tensor &in)
{
    assert(in.shape.size() >= 2);

    Tensor out = Zeros({in.shape.back(), in.shape[in.shape.size() - 2]});

    out.num_ch_dim = 1;

    for (size_t i = 0; i < out.shape.size() - 1; ++i)
        out.num_ch_dim *= out.shape[i];

    std::vector<size_t> idx_rows;

    for (size_t i = 0; i < in.num_ch_dim; ++i)
        idx_rows.push_back(i * in.shape.back());

    size_t batch_size = GetBatchSize(in.shape);

    size_t idx = 0;

    for (size_t i = 0; i < batch_size; ++i)
    {
        for (size_t j = 0; j < out.shape[out.shape.size() - 2]; ++j)
        {
            for (size_t k = 0; k < out.shape.back(); ++k)
            {
                out[idx] = in[idx_rows[k + (i * out.shape.back())]];
                idx_rows[k + (i * out.shape.back())] += 1;
                ++idx;
            }
        }
    }

    return out;
}

Tensor Zeros(const std::vector<size_t> &shape)
{
    Tensor newTensor = Tensor();

    for (const size_t &i : shape)
        assert(i != 0);

    newTensor.shape = shape;

    if (newTensor.shape.size() > 0)
        newTensor.size =
            std::accumulate(newTensor.shape.begin(), newTensor.shape.end(), 1ULL, std::multiplies<size_t>());
    else
        newTensor.size = 1;

    newTensor.elem = new float[newTensor.size];
    std::fill(newTensor.elem, newTensor.elem + newTensor.size, 0.0f);

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