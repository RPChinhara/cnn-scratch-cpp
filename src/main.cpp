#include "lyrs.h"
#include "acts.h"
#include "arrs.h"
#include "datas.h"
#include "linalg.h"
#include "losses.h"
#include "math.hpp"
#include "preproc.h"
#include "rd.h"
#include "tensor.h"

#include <chrono>

class cnn_2d {
  private:
    float lr;
    std::vector<size_t> filters;

    std::vector<tensor> forward(const tensor &input, const std::vector<tensor> &kernel, const size_t stride);

  public:
    cnn_2d();
    void train(const tensor &x_train, const tensor &y_train);
    float evaluate(const tensor &x_test, const tensor &y_test);
    void predict(const tensor &x_test, const tensor &y_test);
};

cnn_2d::cnn_2d() {
}

void cnn_2d::train(const tensor &x_train, const tensor &y_train) {
    tensor kernel = tensor({3, 3}, {1, -1, 1, 0, 1, 0, -1, 0, 1});

    size_t kernelHeight = kernel.shape.front();
    size_t kernelWidth = kernel.shape.back();

    size_t inputHeight = x_train.shape[1];
    size_t inputWidth = x_train.shape[2];

    size_t outputHeight = inputHeight - kernelHeight + 1;
    size_t outputWidth = inputWidth - kernelWidth + 1;

    tensor output = zeros({outputHeight, outputWidth});
}

float cnn_2d::evaluate(const tensor &x_test, const tensor &y_test) {
    return 0.0f;
}

void cnn_2d::predict(const tensor &x_test, const tensor &y_test) {
}

std::vector<tensor> cnn_2d::forward(const tensor &input, const std::vector<tensor> &kernel, const size_t stride) {
    std::vector<tensor> weights;

    return weights;
}

int main() {
    mnist data = load_mnist();

    for (auto i = 0; i < 784; ++i) {
        if (i % 28 == 0)
            std::cout << std::endl;
        std::cout << data.trainImages[i] << "   ";
    }

    data.trainImages / 255.0f;
    data.testImages / 255.0f;

    data.trainLabels = one_hot(data.trainLabels, 10);
    data.testLabels = one_hot(data.testLabels, 10);

    cnn_2d model = cnn_2d();
    model.train(data.trainImages, data.trainLabels);

    return 0;
}