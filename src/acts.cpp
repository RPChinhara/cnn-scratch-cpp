#include "acts.h"
#include "math.hpp"
#include "tensor.h"

tensor hyperbolic_tangent(const tensor &z_t) {
    tensor h_t = z_t;

    for (auto i = 0; i < z_t.size; ++i)
        h_t.elem[i] = std::tanhf(z_t.elem[i]);

    return h_t;
}

tensor sigmoid(const tensor &t) {
    tensor t_new = t;

    return 1.0 / (1.0 + exp(-t));
}