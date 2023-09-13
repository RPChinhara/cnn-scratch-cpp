#include "activations.h"
#include "derivatives.h"
#include "mathematics.h"
#include "tensor.h"

Tensor categorical_crossentropy_prime(const Tensor& y_true, const Tensor& y_pred) {
    return (y_pred - y_true);
}

Tensor l1_prime(const float lambda, const Tensor& w) {
    Tensor out = w;
    for (unsigned int i = 0; i < w._size; ++i) {
        if (w[i] < 0.0f) 
            out[i] = -1.0f;
        else if (w[i] == 0.0f) 
            out[i] = 0.0f;
        else if (w[i] > 1.0f) 
            out[i] = 1.0f;
    }
    return lambda * out;
}

Tensor l2_prime(const float lambda, const Tensor& w) {
    return lambda * w;
}

Tensor mean_squared_error_prime(const Tensor& y_true, const Tensor& y_pred) {
    // 2 / n * (y_pred - y_true) where n is number of columns (features). If it's hard to imagine think if y_true and y_pred had shape (1, some number of features), then it's weird n to be number of rows which is always 1 in this case. In addition to this, reasons why n is size of y_true and y_pred in losses::mse() is that it has to return scaler, also it actually number of columns meaning if y_true and y_pred were matrix which is most of the case it has to take averages of each row (deviding by number of columns), and add all of them up, and devide by number of rows which results n to be size of y_true and y_pred (number of columns * number of rows) after all in mse.
    return (2.0f / y_true._shape.back()) * (y_pred - y_true);
}

Tensor relu_prime(const Tensor& in) {
    Tensor out = in;
    for (unsigned int i = 0; i < in._size; ++i) {
        if (in[i] < 0.0f)
            out[i] = 0.0f;
        else if (in[i] > 0.0f)
            out[i] = 1.0f;
        else if (in[i] == 0.0f)
            out[i] = 0.0f;
    }
    return out;
}

Tensor sigmoid_prime(const Tensor& in) {
    return in * (1.0f - in);
}