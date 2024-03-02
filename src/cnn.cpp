#include "cnn.h"

#include "array.h"
#include "tensor.h"

CNN2D::CNN2D(const std::vector<size_t> &filters, float const learning_rate)
{
    this->filters = filters;
    this->learning_rate = learning_rate;
}

void CNN2D::Train(const Tensor &xTrain, const Tensor &yTrain, const Tensor &xVal, const Tensor &yVal)
{
    Tensor kernel = Zeros({3, 3});

    size_t kernelHeight = kernel.shape.front();
    size_t kernelWidth = kernel.shape.back();

    size_t inputHeight = xTrain.shape[1];
    size_t inputWidth = xTrain.shape[2];
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