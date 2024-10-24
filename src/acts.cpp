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

tensor relu_derivative(const tensor &a) {
    tensor t_new = a;

    for (auto i = 0; i < a.size; ++i)
        t_new[i] = (a[i] > 0.0f) ? 1.0f : 0.0f;

    return t_new;
}