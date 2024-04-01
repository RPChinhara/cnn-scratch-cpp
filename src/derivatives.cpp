#include "derivatives.h"
#include "tensor.h"

Tensor dl_da_da_dz(const Tensor &y_target, const Tensor &y_pred, Act act)
{
    switch (act)
    {
    case SIGMOID: {
        return (y_pred - y_target) * y_pred * (1 - y_pred);
        break;
    }
    case SOFTMAX: {
        return (y_pred - y_target);
        break;
    }
    default:
        std::cout << "Unknown activation." << std::endl;
        return Tensor();
    }
}

Tensor da_dz(const Tensor &tensor, Act act)
{
    Tensor newTensor = tensor;

    switch (act)
    {
    case RELU: {
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
        break;
    }
    default:
        std::cout << "Unknown activation." << std::endl;
        return Tensor();
    }
}