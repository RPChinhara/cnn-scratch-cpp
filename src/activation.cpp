#include "activation.h"
#include "array.h"
#include "mathematics.h"
#include "tensor.h"

Tensor Relu(const Tensor& in)
{
    Tensor zeros = Zeros({ in.shape });
    return Maximum(in, zeros);
}

Tensor Softmax(const Tensor& in)
{
    Tensor exp_scores = Exp(in - Max(in, 1));
    return exp_scores / Sum(exp_scores, 1);
}

Tensor Softplus(const Tensor& in)
{
    Tensor out = in;

    for (size_t i = 0; i < in.size; ++i)
        out[i] = std::logf(std::expf(in[i]) + 1.0f);

    return out;
}