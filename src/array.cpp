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