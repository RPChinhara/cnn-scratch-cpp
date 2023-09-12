#include "preprocessing.h"
#include "mathematics.h"
#include "random.h"

#include <random>

Tensor min_max_scaler(Tensor& dataset) {
    auto min_vals = min(dataset);
    auto max_vals = max(dataset, 0);
    return (dataset - min_vals) / (max_vals - min_vals);
}

Tensor one_hot(const Tensor& in, const u16 depth) {
    Tensor out = Tensor({ 0.0f }, { in._size, depth });

    std::vector<f32> indices;

    // If 'in' was { 0, 1, 2 }, and 'depth' was 3, then this function should return
    // 1 0 0
    // 0 1 0
    // 0 0 1
    //
    // 0 1 2
    // 3 4 5
    // 6 7 8
    //
    // If 'in' was { 0, 0, 0 }, then indices should become { 0, 3, 6 }.
    // If 'in' was { 0, 1, 2 }, then indices should become { 0, 4, 8 }.
    for (u32 i = 0; i < in._size; ++i) {
        if (i == 0)
            indices.push_back(in[i]);
        else
            indices.push_back(in[i] + (i * depth));
    }

    for (u32 i = 0; i < out._size; ++i) {
        for (auto j : indices) {
            if (i == j)
                out[i] = 1.0f;
        }
    }

    return out;
}

// TODO: Check what will happen if test_size given is like 0.21, observe (u32)(std::floorf(x._shape.front() * (1.0 - test_size))).
// TODO: It only works for case when y is one hot encoded.
TrainTest train_test_split(const Tensor x, const Tensor y, const float test_size, const u32 random_state) {
    Tensor new_x = shuffle(x, random_state);
    Tensor new_y = shuffle(y, random_state);

    TrainTest train_test;
    train_test.x_first = Tensor({ 0.0 }, { (u32)(std::floorf(x._shape.front() * (1.0 - test_size))), x._shape.back() });
    train_test.x_second  = Tensor({ 0.0 }, { (u32)(std::floorf(x._shape.front() * test_size)), x._shape.back() });
    train_test.y_first = Tensor({ 0.0 }, { (u32)(std::floorf(y._shape.front() * (1.0 - test_size))), y._shape.back() });
    train_test.y_second  = Tensor({ 0.0 }, { (u32)(std::floorf(y._shape.front() * test_size)), y._shape.back() });

    for (u32 i = 0; i < train_test.x_first._size; ++i)
        train_test.x_first[i] = new_x[i];

    u32 idx = 0;
    for (u32 i = train_test.x_first._size; i < x._size; ++i) {
        train_test.x_second[idx] = new_x[i];
        ++idx;
    }

    for (u32 i = 0; i < train_test.y_first._size; ++i)
        train_test.y_first[i] = new_y[i];

    idx = 0;
    for (u32 i = train_test.y_first._size; i < y._size; ++i) {
        train_test.y_second[idx] = new_y[i];
        ++idx;
    }

    return train_test;
}