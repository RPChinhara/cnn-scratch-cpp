#include "acts.h"
#include "math.hpp"
#include "tensor.h"

tensor hyperbolic_tangent(const tensor& x) {
    tensor y = x;

    for (auto i = 0; i < x.size; ++i)
        y.elems[i] = std::tanhf(x.elems[i]);

    return y;
}

tensor relu(const tensor& x) {
    tensor y = x;

    for (auto i = 0; i < x.size; ++i)
        y.elems[i] = std::fmax(0.0f, x.elems[i]);

    return y;
}

tensor sigmoid(const tensor& x) {
    tensor y = x;

    return 1.0 / (1.0 + exp(-x));
}

tensor softmax(const tensor& x) {
    tensor exp_scores = exp(x - max(x, 1));
    return exp_scores / sum(exp_scores, 1);
}

tensor relu_derivative(const tensor& x) {
    tensor y = x;

    for (auto i = 0; i < x.size; ++i)
        y[i] = (x[i] > 0.0f) ? 1.0f : 0.0f;

    return y;
}

tensor sigmoid_derivative(const tensor& x) {
    tensor y = sigmoid(x);
    return y * (1.0f - y);
}