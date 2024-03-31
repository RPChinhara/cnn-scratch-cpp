#include "derivatives.h"
#include "tensor.h"

Tensor dcce_da_da_dz(const Tensor &yTrue, const Tensor &yPred)
{
    return (yPred - yTrue);
}

Tensor drelu_dz(const Tensor &tensor)
{
    Tensor newTensor = tensor;

    for (size_t i = 0; i < tensor.size; ++i)
    {
        if (tensor[i] > 0.0f)
            newTensor[i] = 1.0f;
        else if (tensor[i] == 0.0f)
            newTensor[i] = 0.0f;
        else
            newTensor[i] = 0.0f;
    }

    return newTensor;
}