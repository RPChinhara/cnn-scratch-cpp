#include "preprocessing.h"
#include "array.h"
#include "mathematics.h"
#include "random.h"

Tensor MinMaxScaler(Tensor& dataset)
{
    auto min_vals = Min(dataset);
    auto max_vals = Max(dataset, 0);
    return (dataset - min_vals) / (max_vals - min_vals);
}

Tensor OneHot(const Tensor& in, const unsigned short depth)
{
    Tensor out = Zeros({ in.size, depth });

    std::vector<float> indices;

    for (unsigned int i = 0; i < in.size; ++i) {
        if (i == 0)
            indices.push_back(in[i]);
        else
            indices.push_back(in[i] + (i * depth));
    }

    for (unsigned int i = 0; i < out.size; ++i) {
        for (auto j : indices) {
            if (i == j)
                out[i] = 1.0f;
        }
    }

    return out;
}

TrainTest TrainTestSplit(const Tensor& x, const Tensor& y, const float test_size, const unsigned int random_state)
{
    Tensor x_new = Shuffle(x, random_state);
    Tensor y_new = Shuffle(y, random_state);

    TrainTest train_test;
    train_test.x_first  = Zeros({ (unsigned int)(std::floorf(x.shape.front() * (1.0 - test_size))), x.shape.back() });
    train_test.x_second = Zeros({ (unsigned int)(std::ceilf(x.shape.front() * test_size)),          x.shape.back() });
    train_test.y_first  = Zeros({ (unsigned int)(std::floorf(y.shape.front() * (1.0 - test_size))), y.shape.back() });
    train_test.y_second = Zeros({ (unsigned int)(std::ceilf(y.shape.front() * test_size)),          y.shape.back() });

    for (unsigned int i = 0; i < train_test.x_first.size; ++i)
        train_test.x_first[i] = x_new[i];

    unsigned int idx = 0;

    for (unsigned int i = train_test.x_first.size; i < x.size; ++i) {
        train_test.x_second[idx] = x_new[i];
        ++idx;
    }

    for (unsigned int i = 0; i < train_test.y_first.size; ++i)
        train_test.y_first[i] = y_new[i];

    idx = 0;

    for (unsigned int i = train_test.y_first.size; i < y.size; ++i) {
        train_test.y_second[idx] = y_new[i];
        ++idx;
    }

    return train_test;
}