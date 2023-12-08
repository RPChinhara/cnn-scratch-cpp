#include "array.h"
#include "tensor.h"

#include <cassert>

Tensor ClipByValue(const Tensor& in, float clip_val_min, float clip_val_max)
{
    assert(clip_val_min <= clip_val_max);
    Tensor out = in;
    
    for (unsigned int i = 0; i < in.size; ++i) {
        if (in[i] < clip_val_min)
            out[i] = clip_val_min;
        else if (in[i] > clip_val_max)
            out[i] = clip_val_max;
    }

    return out;
}

static void SetShape(Tensor& in, const std::vector<size_t>& shape)
{
    in.shape.reserve(shape.size());

    for (unsigned int elem : shape)
        assert(elem != 0);

    in.shape = std::move(shape);
}

static void SetSize(Tensor& in, const std::vector<size_t>& shape)
{
    if (in.shape.size() > 0) {
        unsigned int num_elem = 1;

        for (unsigned int elem : shape)
            num_elem *= elem;

        in.size = num_elem;
    } else {
        in.size = 1;
    }
}

static void SetElem(Tensor& out, const float value)
{
    out.elem = new float[out.size];
    std::fill(out.elem, out.elem + out.size, value);
}

static void SetNumChDim(Tensor& in,  const std::vector<size_t>& shape) 
{
    if (in.shape.size() > 0) {
        in.num_ch_dim = 1;

        for (int i = 0; i < shape.size() - 1; ++i)
            in.num_ch_dim *= shape[i];

    } else {
        in.num_ch_dim = 0;
    }
}

Tensor Ones(const std::vector<size_t>& shape)
{
    Tensor out = Tensor();
    SetShape(out, shape);
    SetSize(out, shape);
    SetElem(out, 1.0f);
    SetNumChDim(out, shape);
    return out;
}

Tensor Slice(const Tensor& in, const unsigned int begin, const unsigned int size)
{
    assert(begin < in.shape[0] && begin + size <= in.shape[0]);
    Tensor out = Zeros({ size, in.shape.back() });
    unsigned int idx = 0;

    for (unsigned int i = begin * in.shape.back(); i < (begin * in.shape.back()) + (size * in.shape.back()) ; ++i) {
        out[idx] = in[i];
        ++idx;
    }

    return out;
}

Tensor Zeros(const std::vector<size_t>& shape) 
{
    Tensor out = Tensor();
    SetShape(out, shape);
    SetSize(out, shape);
    SetElem(out, 0.0f);
    SetNumChDim(out, shape);
    return out;
}