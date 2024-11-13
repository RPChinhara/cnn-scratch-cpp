#include "acts.h"
#include "math.hpp"
#include "tensor.h"

tensor hyperbolic_tangent(const tensor &z_t) {
    tensor h_t = z_t;

    for (auto i = 0; i < z_t.size; ++i)
        h_t.elems[i] = std::tanhf(z_t.elems[i]);

    return h_t;
}

tensor relu(const tensor &z) {
    tensor a = z;

    for (auto i = 0; i < z.size; ++i)
        a.elems[i] = std::fmax(0.0f, z.elems[i]);

    return a;
}

tensor sigmoid(const tensor &t) {
    tensor t_new = t;

    return 1.0 / (1.0 + exp(-t));
}

tensor softmax(const tensor &z) {
    tensor exp_scores = exp(z - max(z, 1));
    return exp_scores / sum(exp_scores, 1);
}

tensor relu_derivative(const tensor &z) {
    tensor t_new = z;

    for (auto i = 0; i < z.size; ++i)
        t_new[i] = (z[i] > 0.0f) ? 1.0f : 0.0f;

    return t_new;
}

tensor sigmoid_derivative(const tensor &z) {
    tensor s = sigmoid(z);
    return s * (1.0f - s);
}