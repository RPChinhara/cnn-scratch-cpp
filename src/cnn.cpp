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

    // # Get image and kernel dimensions
    // img_height, img_width = batch_images.shape[1:3]
    // kernel_height, kernel_width = kernel.shape
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