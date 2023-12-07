#include "activations.h"
#include "derivatives.h"
#include "mathematics.h"
#include "tensor.h"

Tensor PrimeCategoricalCrossEntropy(const Tensor& yTrue, const Tensor& yPred)
{
    return (yPred - yTrue);
}

Tensor PrimeMeanSquaredError(const Tensor& yTrue, const Tensor& yPred)
{
    return (2.0f / yTrue._shape.back()) * (yPred - yTrue);
}

Tensor PrimeRelu(const Tensor& in)
{
    Tensor out = in;

    for (unsigned int i = 0; i < in._size; ++i) {
        if (in[i] < 0.0f)
            out[i] = 0.0f;
        else if (in[i] > 0.0f)
            out[i] = 1.0f;
        else if (in[i] == 0.0f)
            out[i] = 0.0f;
    }

    return out;
}

Tensor PrimeSigmoid(const Tensor& in)
{
    return in * (1.0f - in);
}