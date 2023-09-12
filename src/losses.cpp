#include "losses.h"
#include "arrays.h"
#include "mathematics.h"
#include "tensor.h"

f32 categorical_crossentropy(const Tensor& y_true, const Tensor& y_pred) {
    f32 sum               = 0.0f;
    f32 epsilon           = 1e-15f; // A small value to avoid division by zero
    u32 num_samples       = y_true._shape.front();
    Tensor clipped_y_pred = clip_by_value(y_pred, epsilon, 1.0f - epsilon); // Clip the predicted probabilities to avoid log(0) errors

    for (u32 i = 0; i < y_true._size; ++i)
        sum += y_true[i] * log(clipped_y_pred)[i];
    return -sum / num_samples;
}

f32 mean_squared_error(const Tensor& y_true, const Tensor& y_pred) {
    f32 sum = 0.0f;
    for (u32 i = 0; i < y_true._size; ++i)
        sum += std::powf(y_true[i] - y_pred[i], 2.0f); 
    return sum / y_true._size;
}