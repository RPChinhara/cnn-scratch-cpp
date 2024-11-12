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

class cnn2d {
  private:
    std::vector<size_t> filters;
    float lr;

    std::vector<tensor> forward(const tensor &input, const std::vector<tensor> &kernel, const size_t stride);

  public:
    cnn2d(const std::vector<size_t> &filters, float const lr);
    void train(const tensor &x_train, const tensor &y_train, const tensor &x_val, const tensor &y_val);
    void predict(const tensor &xTest, const tensor &yTest);
};

cnn2d::cnn2d(const std::vector<size_t> &filters, float const lr) {
    this->filters = filters;
    this->lr = lr;
}

void cnn2d::train(const tensor &xTrain, const tensor &yTrain, const tensor &xVal, const tensor &yVal) {
    tensor kernel = tensor({3, 3}, {1, -1, 1, 0, 1, 0, -1, 0, 1});

    size_t kernelHeight = kernel.shape.front();
    size_t kernelWidth = kernel.shape.back();

    size_t inputHeight = xTrain.shape[1];
    size_t inputWidth = xTrain.shape[2];

    size_t outputHeight = inputHeight - kernelHeight + 1;
    size_t outputWidth = inputWidth - kernelWidth + 1;

    tensor output = zeros({outputHeight, outputWidth});
}

void cnn2d::predict(const tensor &xTest, const tensor &yTest) {
}

std::vector<tensor> cnn2d::forward(const tensor &input, const std::vector<tensor> &kernel, const size_t stride) {
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

    cnn2d cnn2D = cnn2d({3, 128, 3}, 0.01f);
    cnn2D.train(data.trainImages, data.trainLabels, data.testImages, data.testLabels);

    return 0;
}