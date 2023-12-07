#include "activations.h"
#include "mathematics.h"
#include "tensor.h"

Tensor Relu(const Tensor& in)
{
    Tensor zeros = Tensor({ 0.0 }, { in._shape });
    return Maximum(in, zeros);
}

Tensor Sigmoid(const Tensor& in)
{
    Tensor out = in;

    for (unsigned int i = 0; i < in._size; ++i)
        out[i] = 1.0f / (1.0f + std::expf(-in[i]));

    return out;
}

Tensor Softmax(const Tensor& in)
{
    Tensor expScores = Exp(in - Max(in, 1));
    return expScores / Sum(expScores, 1);
}

Tensor Softplus(const Tensor& in)
{
    Tensor out = in;

    for (unsigned int i = 0; i < in._size; ++i)
        out[i] = std::logf(std::expf(in[i]) + 1.0f);

    return out;
}