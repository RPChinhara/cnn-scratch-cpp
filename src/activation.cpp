#include "activation.h"
#include "array.h"
#include "mathematics.h"
#include "tensor.h"

Tensor Relu(const Tensor& in)
{
    Tensor zeros = Zeros({ in.shape });
    return Maximum(in, zeros);
}

Tensor Sigmoid(const Tensor& in)
{
    Tensor out = in;

    for (unsigned int i = 0; i < in.size; ++i)
        out[i] = 1.0f / (1.0f + std::expf(-in[i]));

    return out;
}

Tensor Softmax(const Tensor& in)
{
    Tensor exp_scores = Exp(in - Max(in, 1));
    return exp_scores / Sum(exp_scores, 1);
}

Tensor Softplus(const Tensor& in)
{
    Tensor out = in;

    for (unsigned int i = 0; i < in.size; ++i)
        out[i] = std::logf(std::expf(in[i]) + 1.0f);

    return out;
}