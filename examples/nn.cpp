#include "datas.h"
#include "lyrs.h"
#include "math.hpp"
#include "preproc.h"

#include <chrono>

ten relu(const ten &z) {
    ten a = z;

    for (auto i = 0; i < z.size; ++i)
        a.elem[i] = std::fmax(0.0f, z.elem[i]);

    return a;
}

ten softmax(const ten &z) {
    ten exp_scores = exp(z - max(z, 1), CPU);
    return exp_scores / sum(exp_scores, 1);
}

float categorical_cross_entropy(const ten &y_true, const ten &y_pred) {
    float sum = 0.0f;
    constexpr float epsilon = 1e-15f;
    size_t num_samples = y_true.shape.front();
    ten y_pred_clipped = clip_by_value(y_pred, epsilon, 1.0f - epsilon);
    ten y_pred_logged = log(y_pred_clipped, CPU);

    for (auto i = 0; i < y_true.size; ++i)
        sum += y_true[i] * y_pred_logged[i];

    return -sum / num_samples;
}

float categorical_accuracy(const ten &y_true, const ten &y_pred) {
    ten idx_true = argmax(y_true);
    ten pred_idx = argmax(y_pred);
    float equal = 0.0f;

    for (auto i = 0; i < idx_true.size; ++i)
        if (idx_true[i] == pred_idx[i])
            ++equal;

    return equal / idx_true.size;
}

int main() {
    const size_t depth = 3;

    const float test_size1 = 0.2f;
    const float test_size2 = 0.5f;
    const size_t rd_state = 42;

    const size_t num_input = 4;
    const size_t num_hidden1 = 64;
    const size_t num_hidden2 = 64;
    const size_t num_output = 3;
    const float lr = 0.01f;

    iris data = load_iris();
    ten x = data.x;
    ten y = data.y;

    y = one_hot(y, depth);

    auto train_temp = split_dataset(x, y, test_size1, rd_state);
    auto val_test = split_dataset(train_temp.x_test, train_temp.y_test, test_size2, rd_state);

    train_temp.x_train = min_max_scaler(train_temp.x_train);
    val_test.x_train = min_max_scaler(val_test.x_train);
    val_test.x_test = min_max_scaler(val_test.x_test);

    nn model = nn({num_input, num_hidden1, num_hidden2, num_output}, {relu, relu, softmax}, lr, categorical_cross_entropy, categorical_accuracy);

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