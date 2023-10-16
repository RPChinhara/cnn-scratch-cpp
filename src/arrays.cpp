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

Tensor slice(const Tensor& in, const unsigned int begin, const unsigned int size) {
    // Value of begin has to be less than number of row. Also, begin + size must be less or equal to the number of row, otherwise it would try to access a row that dosen't exist which would just return 0s though.
    assert(begin < in._shape[0] && begin + size <= in._shape[0]);
    Tensor out = Tensor( { 0.0f }, { size, in._shape.back() });
    unsigned int idx = 0;
    for (unsigned int i = begin * in._shape.back(); i < (begin * in._shape.back()) + (size * in._shape.back()) ; ++i) {
        out[idx] = in[i];
        ++idx;
    }
    return out;
}