#include "cnn2d.h"
#include "arrs.h"
#include "ten.h"

cnn2d::cnn2d(const std::vector<size_t> &filters, float const learning_rate)
{
    this->filters = filters;
    this->learning_rate = learning_rate;
}

void cnn2d::Train(const ten &xTrain, const ten &yTrain, const ten &xVal, const ten &yVal)
{
    // ten kernel = zeros({3, 3});
    ten kernel = ten({1, -1, 1, 0, 1, 0, -1, 0, 1}, {3, 3});

    size_t kernelHeight = kernel.shape.front();
    size_t kernelWidth = kernel.shape.back();

    size_t inputHeight = xTrain.shape[1];
    size_t inputWidth = xTrain.shape[2];

    size_t outputHeight = inputHeight - kernelHeight + 1;
    size_t outputWidth = inputWidth - kernelWidth + 1;

    ten output = zeros({outputHeight, outputWidth});

    // size_t idx = 0;

    // for (auto i = 0; i < outputHeight; ++i)
    // {
    //     for (auto j = 0; i < outputWidth; ++j)
    //     {
    //         // ouput[idx] =
    //     }
    // }

    // std::cout << output << std::endl;
}

void cnn2d::Predict(const ten &xTest, const ten &yTest)
{
}

std::vector<ten> cnn2d::ForwardPropagation(const ten &input, const std::vector<ten> &kernel, const size_t stride)
{
    std::vector<ten> weights;

    return weights;
}