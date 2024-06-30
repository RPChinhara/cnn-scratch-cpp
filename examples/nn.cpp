#include "datas.h"
#include "lyrs.h"
#include "preproc.h"

#include <chrono>

int main()
{
    const size_t depth = 3;

    const float test_size1 = 0.2f;
    const float test_size2 = 0.5f;
    const size_t rd_state = 42;

    const size_t num_in_neurons = 4;
    const size_t num_hidden1_neurons = 64;
    const size_t num_hidden2_neurons = 64;
    const size_t num_out_neurons = 3;
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

    nn classifier =
        nn({num_in_neurons, num_hidden1_neurons, num_hidden2_neurons, num_out_neurons}, {RELU, RELU, SOFTMAX}, lr);

    auto start = std::chrono::high_resolution_clock::now();

    classifier.train(train_temp.x_train, train_temp.y_train, val_test.x_train, val_test.y_train);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Time taken: " << duration.count() << " seconds\n";

    classifier.pred(val_test.x_test, val_test.y_test);

    return 0;
}