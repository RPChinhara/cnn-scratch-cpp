#include "cnn.h"

#include "tensor.h"

CNN2D::CNN2D(const std::vector<size_t> &filters, float const learning_rate)
{
    this->filters = filters;
    this->learning_rate = learning_rate;
}

void CNN2D::Train()
{
    //     # Convolution operation and its derivative
    // def conv2d(image, kernel, stride=1):
    //     # ...

    // def conv2d_backward(image, kernel, grad_output, stride=1):
    //     # ...

    // # Max pooling operation and its derivative
    // def max_pooling(input_data, pool_size=2, stride=2):
    //     # ...

    // def max_pooling_backward(input_data, pool_size, stride, grad_output):
    //     # ...

    // # Fully connected layer and its derivative
    // def dense(input_data, weights, bias):
    //     # ...

    // def dense_backward(dense_input, weights, bias, grad_output):
    //     # ...

    // # Flatten operation and its derivative
    // def flatten(input_data):
    //     # ...

    // def flatten_backward(flattened_input, original_shape, grad_output):
    //     # ...
}

void CNN2D::Predict()
{
}

std::vector<Tensor> CNN2D::ForwardPropagation(const Tensor &input, const std::vector<Tensor> &kernel,
                                              const size_t stride)
{
    std::vector<Tensor> weights;

    return weights;
}