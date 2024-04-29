#include "diffs.h"
#include "ten.h"

ten da_dz(const ten &t, act_enum act)
{
    ten t_new = t;

    switch (act)
    {
    case RELU: {
        for (auto i = 0; i < t.size; ++i)
        {
            if (t[i] > 0.0f)
                t_new[i] = 1.0f;
            else if (t[i] == 0.0f)
                t_new[i] = 0.0f;
            else
                t_new[i] = 0.0f;
        }

        return t_new;
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
    case SIGMOID: {
        return (y_pred - y_true) * y_pred * (1 - y_pred);
    }
    case SOFTMAX: {
        return (y_pred - y_true);
    }
    default:
        std::cout << "Unknown act." << std::endl;
        return ten();
    }
}