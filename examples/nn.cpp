#include "datas.h"
#include "lyrs.h"
#include "math.hpp"
#include "preproc.h"

#include <chrono>

tensor relu(const tensor &z) {
    tensor a = z;

    for (auto i = 0; i < z.size; ++i)
        a.elems[i] = std::fmax(0.0f, z.elems[i]);

    return a;
}

tensor softmax(const tensor &z) {
    tensor exp_scores = exp(z - max(z, 1));
    return exp_scores / sum(exp_scores, 1);
}

float categorical_cross_entropy(const tensor &y_true, const tensor &y_pred) {
    float sum = 0.0f;
    constexpr float epsilon = 1e-15f;
    size_t num_samples = y_true.shape.front();
    tensor y_pred_clipped = clip_by_value(y_pred, epsilon, 1.0f - epsilon);
    tensor y_pred_logged = log(y_pred_clipped);

    for (auto i = 0; i < y_true.size; ++i)
        sum += y_true[i] * y_pred_logged[i];

    return -sum / num_samples;
}

float categorical_accuracy(const tensor &y_true, const tensor &y_pred) {
    tensor idx_true = argmax(y_true);
    tensor pred_idx = argmax(y_pred);
    float equal = 0.0f;

    for (auto i = 0; i < idx_true.size; ++i)
        if (idx_true[i] == pred_idx[i])
            ++equal;

    return equal / idx_true.size;
}

int main() {
    iris data = load_iris();
    tensor x = data.x;
    tensor y = data.y;

    y = one_hot(y, 3);

    auto train_temp = split_dataset(x, y, 0.2f, 42);
    auto val_test = split_dataset(train_temp.x_test, train_temp.y_test, 0.5f, 42);

    train_temp.x_train = min_max_scaler(train_temp.x_train);
    val_test.x_train = min_max_scaler(val_test.x_train);
    val_test.x_test = min_max_scaler(val_test.x_test);

    nn model = nn({4, 64, 64, 3}, {relu, relu, softmax}, categorical_cross_entropy, categorical_accuracy, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    model.train(train_temp.x_train, train_temp.y_train, val_test.x_train, val_test.y_train);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << std::endl << "Time taken: " << duration.count() << " seconds" << std::endl << std::endl;

    auto train_loss = model.evaluate(train_temp.x_train, train_temp.y_train);
    auto test_loss = model.evaluate(val_test.x_test, val_test.y_test);
    auto pred = model.predict(val_test.x_test);

    std::cout << "Train loss: " << train_loss << std::endl;
    std::cout << "Test  loss: " << test_loss << std::endl;
    std::cout << std::endl << pred << std::endl;
    std::cout << std::endl << val_test.y_test << std::endl;

    return 0;
}