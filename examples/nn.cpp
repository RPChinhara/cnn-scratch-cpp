#include "datas.h"
#include "lyrs.h"
#include "preproc.h"

#include <chrono>

int main()
{
    iris data = load_iris();
    ten x = data.x;
    ten y = data.y;

    y = one_hot(y, 3);

    train_test train_temp = split_dataset(x, y, 0.2, 42);
    train_test val_test = split_dataset(train_temp.x_test, train_temp.y_test, 0.5, 42);

    train_temp.x_train = min_max_scaler(train_temp.x_train);
    val_test.x_train = min_max_scaler(val_test.x_train);
    val_test.x_test = min_max_scaler(val_test.x_test);

    nn classifier = nn({4, 64, 64, 3}, {RELU, RELU, SOFTMAX}, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    classifier.train(train_temp.x_train, train_temp.y_train, val_test.x_train, val_test.y_train);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Time taken: " << duration.count() << " seconds\n";

    classifier.pred(val_test.x_test, val_test.y_test);

    return 0;
}