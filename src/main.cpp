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
    float lr;

    tensor conv1_kernel;
    tensor conv2_kernel;

    tensor fc1_w;
    tensor fc1_b;

    tensor fc2_w;
    tensor fc2_b;

    std::vector<tensor> forward(const tensor &x, const std::vector<tensor> &kernel, const size_t stride);

  public:
    cnn2d();
    void train(const tensor &x_train, const tensor &y_train);
    float evaluate(const tensor &x_test, const tensor &y_test);
    void predict(const tensor &x_test, const tensor &y_test);
};

cnn2d::cnn2d() {
    conv1_kernel = normal_dist({3, 3});
    conv2_kernel = normal_dist({3, 3});

    fc1_w = normal_dist({32 * 7 * 7});
    fc1_b = zeros({1, 1});

    fc2_w = normal_dist({128});
    fc2_b = zeros({1, 1});
}

void cnn2d::train(const tensor &x_train, const tensor &y_train) {
}

float cnn2d::evaluate(const tensor &x_test, const tensor &y_test) {
    return 0.0f;
}

void cnn2d::predict(const tensor &x_test, const tensor &y_test) {
}

std::vector<tensor> cnn2d::forward(const tensor &x, const std::vector<tensor> &kernel, const size_t stride) {
    std::vector<tensor> weights;

    return weights;
}

int main() {
    mnist data = load_mnist();

    for (auto i = 0; i < 784; ++i) {
        if (i % 28 == 0)
            std::cout << std::endl;
        std::cout << data.trainImages[i] << " ";
    }

    data.trainImages / 255.0f;
    data.testImages / 255.0f;

    data.trainLabels = one_hot(data.trainLabels, 10);
    data.testLabels = one_hot(data.testLabels, 10);

    cnn2d model = cnn2d();
    model.train(data.trainImages, data.trainLabels);

    return 0;
}