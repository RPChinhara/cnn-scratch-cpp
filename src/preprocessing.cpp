#include "preprocessing.h"
#include "arrays.h"
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
    train_test.trainFeatures =
        Zeros({static_cast<size_t>(std::floorf(x.shape.front() * (1.0 - test_size))), x.shape.back()});
    train_test.trainTargets =
        Zeros({static_cast<size_t>(std::floorf(y.shape.front() * (1.0 - test_size))), y.shape.back()});
    train_test.testFeatures = Zeros({static_cast<size_t>(std::ceilf(x.shape.front() * test_size)), x.shape.back()});
    train_test.testTargets = Zeros({static_cast<size_t>(std::ceilf(y.shape.front() * test_size)), y.shape.back()});

    for (size_t i = 0; i < train_test.trainFeatures.size; ++i)
        train_test.trainFeatures[i] = x_new[i];

    for (size_t i = 0; i < train_test.trainTargets.size; ++i)
        train_test.trainTargets[i] = y_new[i];

    for (size_t i = train_test.trainFeatures.size; i < x.size; ++i)
        train_test.testFeatures[i - train_test.trainFeatures.size] = x_new[i];

    for (size_t i = train_test.trainTargets.size; i < y.size; ++i)
        train_test.testTargets[i - train_test.trainTargets.size] = y_new[i];

    return train_test;
}