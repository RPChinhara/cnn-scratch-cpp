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
    size_t batch_size;
    size_t epochs = 150;

    tensor conv1_kernel;
    tensor conv2_kernel;

    tensor fc1_w;
    tensor fc1_b;

    tensor fc2_w;
    tensor fc2_b;

    std::vector<tensor> forward(const tensor &x);

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
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        // auto = forward();

        float error = 0.0f;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << error << std::endl;
    }
}

float cnn2d::evaluate(const tensor &x_test, const tensor &y_test) {
    return 0.0f;
}

void cnn2d::predict(const tensor &x_test, const tensor &y_test) {
}

std::vector<tensor> cnn2d::forward(const tensor &x) {
    std::vector<tensor> weights;

    return weights;
}

int main() {
    mnist data = load_mnist();

    for (auto i = 0; i < 784 * 3; ++i) {
        if (i % 28 == 0 && i % 783 != 0)
            std::cout << std::endl;
        std::cout << data.trainImages[i] << " ";

        if (i % 783 == 0 && i != 0)
            std::cout << std::endl;
    }

    for (auto i = 0; i < data.trainImages.size; ++i)
        data.trainImages[i] /= 255.0f;

    for (auto i = 0; i < data.testImages.size; ++i)
        data.testImages[i] /= 255.0f;

    data.trainImages.reshape({60000, 28, 28, 1});
    data.testImages.reshape({10000, 28, 28, 1});

    data.trainLabels = one_hot(data.trainLabels, 10);
    data.testLabels = one_hot(data.testLabels, 10);

    cnn2d model = cnn2d();
    model.train(data.trainImages, data.trainLabels);

    return 0;
}