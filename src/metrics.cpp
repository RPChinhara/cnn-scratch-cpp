#include "metrics.h"
#include "mathematics.h"
#include "tensor.h"

f32 accuracy(const Tensor& y_true, const Tensor& y_pred) {
    f32 count = 0.0f;
    for (u32 i = 0; i < y_true._size; ++i)
        if (std::fabs(y_true[i] - y_pred[i]) < 1e-6f) 
            ++count;
    return count / y_true._size;
}

f32 categorical_accuracy(const Tensor& y_true, const Tensor& y_pred) {
    Tensor true_idx = argmax(y_true);
    Tensor pred_idx = argmax(y_pred);
    f32 equal       = 0.0f;
    for (u32 i = 0; i < true_idx._size; ++i) {
        if (true_idx[i] == pred_idx[i])
            ++equal;
    }
    return equal / true_idx._size;
}