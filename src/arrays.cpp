#include "arrays.h"
#include "tensor.h"

#include <cassert>

Tensor clip_by_value(const Tensor& in, float clip_val_min, float clip_val_max) {
    assert(clip_val_min <= clip_val_max);
    Tensor out = in;
    
    for (unsigned int i = 0; i < in._size; ++i) {
        if (in[i] < clip_val_min)
            out[i] = clip_val_min;
        else if (in[i] > clip_val_max)
            out[i] = clip_val_max;
    }

    return out;
}

static void set_shape(Tensor& in, const std::vector<unsigned int>& shape) {
    in._shape.reserve(shape.size());

    for (unsigned int elem : shape)
        assert(elem != 0);

    in._shape = std::move(shape);
}

static void set_size(Tensor& in, const std::vector<unsigned int>& shape) {
    if (in._shape.size() > 0) {
        unsigned int num_elem = 1;

        for (unsigned int elem : shape)
            num_elem *= elem;

        in._size = num_elem;
    } else {
        in._size = 1;
    }
}

static void set_elem(Tensor& out, const float value) {
    out._elem = new float[out._size];
    std::fill(out._elem, out._elem + out._size, value);
}

static void set_num_ch_dim(Tensor& in,  const std::vector<unsigned int>& shape) {
    if (in._shape.size() > 0) {
        in._num_ch_dim = 1;

        for (int i = 0; i < shape.size() - 1; ++i)
            in._num_ch_dim *= shape[i];

    } else {
        in._num_ch_dim = 0;
    }
}

Tensor ones(const std::vector<unsigned int>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    set_elem(out, 1.0f);
    set_num_ch_dim(out, shape);
    return out;
}

Tensor slice(const Tensor& in, const unsigned int begin, const unsigned int size) {
    assert(begin < in._shape[0] && begin + size <= in._shape[0]);
    Tensor out = Tensor( { 0.0f }, { size, in._shape.back() });
    unsigned int idx = 0;

    for (unsigned int i = begin * in._shape.back(); i < (begin * in._shape.back()) + (size * in._shape.back()) ; ++i) {
        out[idx] = in[i];
        ++idx;
    }

    return out;
}

Tensor zeros(const std::vector<unsigned int>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    set_elem(out, 0.0f);
    set_num_ch_dim(out, shape);
    return out;
}