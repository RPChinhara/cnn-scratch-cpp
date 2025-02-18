#include "losses.h"
#include "arrs.h"
#include "tensor.h"

// TODO: what is CategoricalFocalCrossentropy() in TF?
float categorical_cross_entropy(const tensor& y_true, const tensor& y_pred) {
    float batch_size = static_cast<float>(y_true.shape.front());
    float num_classes = static_cast<float>(y_true.shape.back());
    float loss = 0.0f;

    tensor y_pred_clipped = clip_by_value(y_pred, 1e-12f, 1.0f);

    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j <num_classes; ++j) {
            std::cout << j << "\n";
            if (y_true(i, j) == 1.0f) {
                loss -= log(y_pred_clipped(i, j));
                break;
            }
        }
    }

    return loss / batch_size;
}

float sparse_categorical_cross_entropy(const tensor& y_true, const tensor& y_pred) {
    float batch_size = static_cast<float>(y_true.size);
    float loss = 0.0f;

    for (auto i = 0; i < batch_size; ++i) {
        float true_class = y_true[i];
        loss -= log(y_pred(i, true_class));
    }

    return loss / batch_size;
}

float mean_squared_error(const tensor& y_true, const tensor& y_pred) {
    float sum = 0.0f; // TODO: should be loss?
    float num_elm = static_cast<float>(y_true.shape.front()); // TODO: should be batch_size?

    for (auto i = 0; i < y_true.size; ++i)
        sum += std::powf(y_true[i] - y_pred[i], 2.0f);

    return sum / num_elm;
}