#include "acts.h"
#include "math.hpp"
#include "tensor.h"

tensor hyperbolic_tangent(const tensor& x) {
    tensor y = x;

    for (auto i = 0; i < x.size; ++i)
        y.elems[i] = std::tanhf(x.elems[i]);

    return y;
}

tensor relu(const tensor& z) {
    tensor a = z;

    for (auto i = 0; i < z.size; ++i)
        a.elems[i] = std::fmax(0.0f, z.elems[i]);

    return a;
}

tensor sigmoid(const tensor& t) {
    tensor t_new = t;

    return 1.0 / (1.0 + exp(-t));
}

tensor softmax(const tensor& z) {
    tensor exp_scores = exp(z - max(z, 1));
    return exp_scores / sum(exp_scores, 1);
}

tensor relu_derivative(const tensor& z) {
    tensor t_new = z;

    for (auto i = 0; i < z.size; ++i)
        t_new[i] = (z[i] > 0.0f) ? 1.0f : 0.0f;

    return t_new;
}

tensor sigmoid_derivative(const tensor& z) {
    tensor s = sigmoid(z);
    return s * (1.0f - s);
}