#include "cnn2d.h"
#include "arrs.h"
#include "ten.h"

CNN2D::CNN2D(const std::vector<size_t> &filters, float const learning_rate)
{
    this->filters = filters;
    this->learning_rate = learning_rate;
}

void CNN2D::Train(const Ten &xTrain, const Ten &yTrain, const Ten &xVal, const Ten &yVal)
{
    // Ten kernel = zeros({3, 3});
    Ten kernel = Ten({1, -1, 1, 0, 1, 0, -1, 0, 1}, {3, 3});

    size_t kernelHeight = kernel.shape.front();
    size_t kernelWidth = kernel.shape.back();

    size_t inputHeight = xTrain.shape[1];
    size_t inputWidth = xTrain.shape[2];

    size_t outputHeight = inputHeight - kernelHeight + 1;
    size_t outputWidth = inputWidth - kernelWidth + 1;

    Ten output = zeros({outputHeight, outputWidth});

    // size_t idx = 0;

    // for (size_t i = 0; i < outputHeight; ++i)
    // {
    //     for (size_t j = 0; i < outputWidth; ++j)
    //     {
    //         // ouput[idx] =
    //     }
    // }

    // std::cout << output << std::endl;
}

void CNN2D::Predict(const Ten &xTest, const Ten &yTest)
{
}

std::vector<Ten> CNN2D::ForwardPropagation(const Ten &input, const std::vector<Ten> &kernel, const size_t stride)
{
    std::vector<Ten> weights;

    return weights;
}