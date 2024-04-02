#include "diffs.h"
#include "ten.h"

Ten da_dz(const Ten &ten, Act act)
{
    Ten newTensor = ten;

    switch (act)
    {
    case RELU: {
        for (size_t i = 0; i < ten.size; ++i)
        {
            if (ten[i] > 0.0f)
                newTensor[i] = 1.0f;
            else if (ten[i] == 0.0f)
                newTensor[i] = 0.0f;
            else
                newTensor[i] = 0.0f;
        }

        return newTensor;
        break;
    }
    default:
        std::cout << "Unknown act." << std::endl;
        return Ten();
    }
}

Ten dl_da_da_dz(const Ten &y_target, const Ten &y_pred, Act act)
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
        std::cout << "Unknown act." << std::endl;
        return Ten();
    }
}