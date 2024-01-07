#include "array.h"
#include "tensor.h"

#include <cassert>

Tensor ClipByValue(const Tensor& in, float clip_val_min, float clip_val_max)
{
    assert(clip_val_min <= clip_val_max);
    Tensor out = in;
    
    for (size_t i = 0; i < in.size; ++i) {
        if (in[i] < clip_val_min)
            out[i] = clip_val_min;
        else if (in[i] > clip_val_max)
            out[i] = clip_val_max;
    }

    return out;
}

Tensor Slice(const Tensor& in, const size_t begin, const size_t size)
{
    assert(begin < in.shape[0] && begin + size <= in.shape[0]);
    Tensor out = Zeros({ size, in.shape.back() });
    size_t idx = 0;

    for (size_t i = begin * in.shape.back(); i < (begin * in.shape.back()) + (size * in.shape.back()) ; ++i) {
        out[idx] = in[i];
        ++idx;
    }

    return out;
}

static size_t GetBatchSize(const std::vector<size_t>& shape)
{
    assert(shape.size() > 1);
    size_t batch_size = 1;

    for (size_t i = 0; i < shape.size() - 2; ++i)
        batch_size *= shape[i];
    
    return batch_size;
}

Tensor Transpose(const Tensor& in)
{
    assert(in.shape.size() >= 2);

    Tensor out = Zeros({ in.shape.back(), in.shape[in.shape.size() - 2] });

    out.num_ch_dim = 1;

    for (size_t i = 0; i < out.shape.size() - 1; ++i)
        out.num_ch_dim *= out.shape[i];
    
    std::vector<size_t> idx_rows;
    
    for (size_t i = 0; i < in.num_ch_dim; ++i)
        idx_rows.push_back(i * in.shape.back());

    size_t batch_size = GetBatchSize(in.shape);

    size_t idx = 0;

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < out.shape[out.shape.size() - 2]; ++j) {
            for (size_t k = 0; k < out.shape.back(); ++k) {
                out[idx] = in[idx_rows[k + (i * out.shape.back())]];
                idx_rows[k + (i * out.shape.back())] += 1;
                ++idx;
            }
        }
    }
    
	return out;
}

Tensor Zeros(const std::vector<size_t>& shape) 
{
    Tensor out = Tensor();

    out.shape.reserve(shape.size());

    for (const size_t& i : shape)
        assert(i != 0);

    out.shape = shape;

    if (out.shape.size() > 0) {
        size_t num_elem = 1;

        for (const size_t& i : shape)
            num_elem *= i;

        out.size = num_elem;
    } else {
        out.size = 1;
    }

    out.elem = new float[out.size];
    std::fill(out.elem, out.elem + out.size, 0.0f);

    if (out.shape.size() > 0) {
        out.num_ch_dim = 1;

        for (const size_t& i : shape)
            out.num_ch_dim *= i;
    } else {
        out.num_ch_dim = 0;
    }

    return out;
}