#include "derivatives.h"
#include "tensor.h"

Tensor CategoricalCrossEntropyDerivative(const Tensor &yTrue, const Tensor &yPred)
{
    return (yPred - yTrue);
}

Tensor ReluDerivative(const Tensor &tensor)
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