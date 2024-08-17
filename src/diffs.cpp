#include "diffs.h"
#include "tensor.h"

tensor da_dz(const tensor &a) {
    tensor t_new = a;

    for (auto i = 0; i < a.size; ++i)
    {
        if (0.0f < a[i])
            t_new[i] = 1.0f;
        else if (a[i] == 0.0f)
            t_new[i] = 0.0f;
        else
            t_new[i] = 0.0f;
    }

    return t_new;
}

tensor dl_da_da_dz(const tensor &y_true, const tensor &y_pred) {
    return (y_pred - y_true);
}