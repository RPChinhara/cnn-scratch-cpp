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