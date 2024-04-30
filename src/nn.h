#pragma once

#include "act.h"

#include <vector>

class ten;

class nn
{
  public:
    nn(const std::vector<size_t> &lyrs, const std::vector<act_enum> &act_types, float const lr);
    void train(const ten &x_train, const ten &y_train, const ten &x_val, const ten &y_val);
    void pred(const ten &x_test, const ten &y_test);

  private:
    std::pair<std::vector<ten>, std::vector<ten>> init_params();
    std::vector<ten> forward_prop(const ten &x, const std::vector<ten> &w, const std::vector<ten> &b);

    std::vector<size_t> lyrs;
    std::vector<act_enum> act_types;
    std::pair<std::vector<ten>, std::vector<ten>> w_b;
    std::pair<std::vector<ten>, std::vector<ten>> w_b_mom;
    std::vector<ten> a;
    size_t batch_size = 10;
    size_t epochs = 200;
    float lr;
    float grad_clip_threshold = 8.0f;
    float mom = 0.1f;
    size_t patience = 4;
};

/*
#include "iris.h"
#include "nn.h"
#include "preproc.h"

#include <chrono>

int main()
{
    iris data = load_iris();
    ten x = data.x;
    ten y = data.y;

    y = one_hot(y, 3);

    train_test train_test = train_test_split(x, y, 0.2, 42);
    train_test val_test = train_test_split(train_test.x_test, train_test.y_test, 0.5, 42);

    train_test.x_train = min_max_scaler(train_test.x_train);
    val_test.x_train = min_max_scaler(val_test.x_train);
    val_test.x_test = min_max_scaler(val_test.x_test);

    nn classifier = nn({4, 64, 64, 3}, {RELU, RELU, SOFTMAX}, 0.01f);

    auto start = std::chrono::high_resolution_clock::now();

    classifier.train(train_test.x_train, train_test.y_train, val_test.x_train, val_test.y_train);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

    std::cout << "Time taken: " << duration.count() << " seconds\n";

    classifier.pred(val_test.x_test, val_test.y_test);

    return 0;
}
*/