#include "array.h"
#include "mathematics.h"
#include "tensor.h"

Tensor Relu(const Tensor& in)
{
    return Maximum(in, Zeros({ in.shape }));
}

Tensor Softmax(const Tensor& in)
{
    Tensor exp_scores = Exp(in - Max(in, 1));
    return exp_scores / Sum(exp_scores, 1);
}