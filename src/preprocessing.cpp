#include "preprocessing.h"
#include "mathematics.h"
#include "random.h"

Tensor MinMaxScaler(Tensor& dataset)
{
    auto minVals = Min(dataset);
    auto maxVals = Max(dataset, 0);
    return (dataset - minVals) / (maxVals - minVals);
}

Tensor OneHot(const Tensor& in, const unsigned short depth)
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

TrainTest TrainTestSplit(const Tensor& x, const Tensor& y, const float testSize, const unsigned int randomState)
{
    Tensor xNew = Shuffle(x, randomState);
    Tensor yNew = Shuffle(y, randomState);

    TrainTest trainTest;
    trainTest.xFirst  = Tensor({ 0.0 }, { (unsigned int)(std::floorf(x._shape.front() * (1.0 - testSize))), x._shape.back() });
    trainTest.xSecond = Tensor({ 0.0 }, { (unsigned int)(std::ceilf(x._shape.front() * testSize)),          x._shape.back() });
    trainTest.yFirst  = Tensor({ 0.0 }, { (unsigned int)(std::floorf(y._shape.front() * (1.0 - testSize))), y._shape.back() });
    trainTest.ySecond = Tensor({ 0.0 }, { (unsigned int)(std::ceilf(y._shape.front() * testSize)),          y._shape.back() });

    for (unsigned int i = 0; i < trainTest.xFirst._size; ++i)
        trainTest.xFirst[i] = xNew[i];

    unsigned int idx = 0;

    for (unsigned int i = trainTest.xFirst._size; i < x._size; ++i) {
        trainTest.xSecond[idx] = xNew[i];
        ++idx;
    }

    for (unsigned int i = 0; i < trainTest.yFirst._size; ++i)
        trainTest.yFirst[i] = yNew[i];

    idx = 0;

    for (unsigned int i = trainTest.yFirst._size; i < y._size; ++i) {
        trainTest.ySecond[idx] = yNew[i];
        ++idx;
    }

    return trainTest;
}