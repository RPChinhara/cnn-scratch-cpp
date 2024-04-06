#include "diffs.h"
#include "ten.h"

ten da_dz(const ten &t, Act act)
{
    ten newTensor = t;

    switch (act)
    {
    case RELU: {
        for (size_t i = 0; i < t.size; ++i)
        {
            if (t[i] > 0.0f)
                newTensor[i] = 1.0f;
            else if (t[i] == 0.0f)
                newTensor[i] = 0.0f;
            else
                newTensor[i] = 0.0f;
        }

        return newTensor;
    }
    default:
        std::cout << "Unknown act." << std::endl;
        return ten();
    }
}

ten dl_da_da_dz(const ten &y_target, const ten &y_pred, Act act)
{
    switch (act)
    {
    case SIGMOID: {
        return (y_pred - y_target) * y_pred * (1 - y_pred);
    }
    case SOFTMAX: {
        return (y_pred - y_target);
    }
    default:
        std::cout << "Unknown act." << std::endl;
        return ten();
    }
}