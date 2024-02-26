#include "preprocessing.h"
#include "array.h"
#include "mathematics.h"
#include "random.h"

Tensor MinMaxScaler(Tensor &dataset)
{
    auto min_vals = Min(dataset);
    auto max_vals = Max(dataset, 0);
    return (dataset - min_vals) / (max_vals - min_vals);
}

Tensor OneHot(const Tensor &tensor, const size_t depth)
{
    Tensor newTensor = Zeros({tensor.size, depth});

    std::vector<float> indices;

    for (size_t i = 0; i < tensor.size; ++i)
    {
        if (i == 0)
            indices.push_back(tensor[i]);
        else
            indices.push_back(tensor[i] + (i * depth));
    }

    for (size_t i = 0; i < newTensor.size; ++i)
    {
        for (auto j : indices)
        {
            if (i == j)
                newTensor[i] = 1.0f;
        }
    }

    return newTensor;
}

TrainTest TrainTestSplit(const Tensor &x, const Tensor &y, const float test_size, const size_t random_state)
{
    Tensor x_new = Shuffle(x, random_state);
    Tensor y_new = Shuffle(y, random_state);

    TrainTest train_test;
    train_test.featuresFirst =
        Zeros({static_cast<size_t>(std::floorf(x.shape.front() * (1.0 - test_size))), x.shape.back()});
    train_test.targetsFirst =
        Zeros({static_cast<size_t>(std::floorf(y.shape.front() * (1.0 - test_size))), y.shape.back()});
    train_test.featuresSecond = Zeros({static_cast<size_t>(std::ceilf(x.shape.front() * test_size)), x.shape.back()});
    train_test.targetsSecond = Zeros({static_cast<size_t>(std::ceilf(y.shape.front() * test_size)), y.shape.back()});

    for (size_t i = 0; i < train_test.featuresFirst.size; ++i)
        train_test.featuresFirst[i] = x_new[i];

    for (size_t i = 0; i < train_test.targetsFirst.size; ++i)
        train_test.targetsFirst[i] = y_new[i];

    for (size_t i = train_test.featuresFirst.size; i < x.size; ++i)
        train_test.featuresSecond[i - train_test.featuresFirst.size] = x_new[i];

    for (size_t i = train_test.targetsFirst.size; i < y.size; ++i)
        train_test.targetsSecond[i - train_test.targetsFirst.size] = y_new[i];

    return train_test;
}