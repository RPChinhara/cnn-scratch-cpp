#include "acts.h"
#include "datas.h"
#include "losses.h"
#include "lyrs.h"
#include "math.hpp"
#include "preproc.h"

#include <chrono>

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

    min_max_scaler scaler;
    scaler.fit(x);
    tensor scaled_x = scaler.transform(x);

    y = one_hot(y, 3);

    auto train_temp = split_dataset(scaled_x, y, 0.2f, 42);
    auto val_test = split_dataset(train_temp.x_test, train_temp.y_test, 0.5f, 42);

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