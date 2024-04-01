#include "cnn2d.h"
#include "arrs.h"
#include "tensor.h"

CNN2D::CNN2D(const std::vector<size_t> &filters, float const learning_rate)
{
    this->filters = filters;
    this->learning_rate = learning_rate;
}

void CNN2D::Train(const Tensor &xTrain, const Tensor &yTrain, const Tensor &xVal, const Tensor &yVal)
{
    // Tensor kernel = Zeros({3, 3});
    Tensor kernel = Tensor({1, -1, 1, 0, 1, 0, -1, 0, 1}, {3, 3});

    size_t kernelHeight = kernel.shape.front();
    size_t kernelWidth = kernel.shape.back();

    size_t inputHeight = xTrain.shape[1];
    size_t inputWidth = xTrain.shape[2];

    size_t outputHeight = inputHeight - kernelHeight + 1;
    size_t outputWidth = inputWidth - kernelWidth + 1;

    Tensor output = Zeros({outputHeight, outputWidth});

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

void CNN2D::Predict(const Tensor &xTest, const Tensor &yTest)
{
}

std::vector<Tensor> CNN2D::ForwardPropagation(const Tensor &input, const std::vector<Tensor> &kernel,
                                              const size_t stride)
{
    std::vector<Tensor> weights;

    return weights;
}