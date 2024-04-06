#include "diffs.h"
#include "ten.h"

ten da_dz(const ten &t, act_enum act)
{
    ten newTensor = t;

    switch (act)
    {
    case ACT_RELU: {
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

ten dl_da_da_dz(const ten &y_true, const ten &y_pred, act_enum act)
{
    switch (act)
    {
    case ACT_SIGMOID: {
        return (y_pred - y_true) * y_pred * (1 - y_pred);
    }
    case ACT_SOFTMAX: {
        return (y_pred - y_true);
    }
    default:
        std::cout << "Unknown act." << std::endl;
        return ten();
    }
}