#include "preprocessing.h"
#include "mathematics.h"
#include "random.h"

#include <random>

Tensor min_max_scaler(Tensor& dataset)
{
    auto min_vals = Min(dataset);
    auto max_vals = Max(dataset, 0);
    return (dataset - min_vals) / (max_vals - min_vals);
}

Tensor one_hot(const Tensor& in, const unsigned short depth)
{
    Tensor out = Tensor({ 0.0f }, { in._size, depth });

    std::vector<float> indices;

    for (unsigned int i = 0; i < in._size; ++i) {
        if (i == 0)
            indices.push_back(in[i]);
        else
            indices.push_back(in[i] + (i * depth));
    }

    for (unsigned int i = 0; i < out._size; ++i) {
        for (auto j : indices) {
            if (i == j)
                out[i] = 1.0f;
        }
    }

    return out;
}

TrainTest train_test_split(const Tensor& x, const Tensor& y, const float test_size, const unsigned int random_state)
{
    Tensor new_x = shuffle(x, random_state);
    Tensor new_y = shuffle(y, random_state);

    TrainTest train_test;
    train_test.x_first  = Tensor({ 0.0 }, { (unsigned int)(std::floorf(x._shape.front() * (1.0 - test_size))), x._shape.back() });
    train_test.x_second = Tensor({ 0.0 }, { (unsigned int)(std::ceilf(x._shape.front() * test_size)),          x._shape.back() });
    train_test.y_first  = Tensor({ 0.0 }, { (unsigned int)(std::floorf(y._shape.front() * (1.0 - test_size))), y._shape.back() });
    train_test.y_second = Tensor({ 0.0 }, { (unsigned int)(std::ceilf(y._shape.front() * test_size)),          y._shape.back() });

    for (unsigned int i = 0; i < train_test.x_first._size; ++i)
        train_test.x_first[i] = new_x[i];

    unsigned int idx = 0;

    for (unsigned int i = train_test.x_first._size; i < x._size; ++i) {
        train_test.x_second[idx] = new_x[i];
        ++idx;
    }

    for (unsigned int i = 0; i < train_test.y_first._size; ++i)
        train_test.y_first[i] = new_y[i];

    idx = 0;

    for (unsigned int i = train_test.y_first._size; i < y._size; ++i) {
        train_test.y_second[idx] = new_y[i];
        ++idx;
    }

    return train_test;
}