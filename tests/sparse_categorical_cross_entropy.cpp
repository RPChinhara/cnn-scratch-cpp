#include "arrs.h"
#include "losses.h"
#include "tensor.h"

int main() {
    tensor y_true = variable({1, 2}, {1.0f, 2.0f});
    tensor y_pred = variable({2, 3}, {0.05f, 0.95f, 0.0f, 0.1f, 0.8f, 0.1f});

    std::cout << sparse_categorical_cross_entropy(y_true, y_pred) << "\n";

    return 0;
}