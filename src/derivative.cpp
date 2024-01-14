#include "tensor.h"

Tensor CategoricalCrossEntropyDerivative(const Tensor& y_true, const Tensor& y_pred)
{
    return (y_pred - y_true);
}

Tensor ReluDerivative(const Tensor& in)
{
    Tensor out = in;

    for (size_t i = 0; i < in.size; ++i) {
        if (in[i] < 0.0f)
            out[i] = 0.0f;
        else if (in[i] > 0.0f)
            out[i] = 1.0f;
        else if (in[i] == 0.0f)
            out[i] = 0.0f;
    }

    return out;
}