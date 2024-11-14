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

constexpr float lr = 0.01f;
constexpr size_t batch_size = 32;
constexpr size_t epochs = 5;

tensor conv1_kernel = normal_dist({3, 3});
tensor conv2_kernel = normal_dist({3, 3});

tensor fc1_w = normal_dist({32 * 7 * 7});
tensor fc1_b = zeros({1, 1});

tensor fc2_w = normal_dist({128});
tensor fc2_b = zeros({1, 1});

tensor cnn2d_convolution(const tensor &x, const tensor &kernel) {

    return tensor();
}

tensor cnn2d_max_pool(const tensor &x) {

    return tensor();
}

tensor cnn2d_forward(const tensor &x) {
    auto x_conv1 = cnn2d_convolution(x, conv1_kernel);
    x_conv1 = relu(x_conv1);
    x_conv1 = cnn2d_max_pool(x_conv1);

    auto x_conv2 = cnn2d_convolution(x_conv1, conv2_kernel);
    x_conv2 = relu(x_conv2);
    x_conv2 = cnn2d_max_pool(x_conv2);

    auto x_fc = matmul(fc1_w, x_conv2) + fc1_b;
    x_fc = matmul(fc2_w, x_fc) + fc2_b;

    return x_fc;
}

void cnn2d_train(const tensor &x_train, const tensor &y_train) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto y = cnn2d_forward(x_train);

        float error = 0.0f;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << error << std::endl;
    }
}

float cnn2d_evaluate(const tensor &x_test, const tensor &y_test) {
    return 0.0f;
}

void cnn2d_predict(const tensor &x_test, const tensor &y_test) {
}

int main() {
    mnist data = load_mnist();

    constexpr size_t num_digits = 10;
    constexpr size_t image_size = 784;
    constexpr size_t image_dim = 28;

    for (auto i = 0; i < num_digits; ++i) {
        for (auto j = 0; j < image_size; ++j) {
            if (j % image_dim == 0 && j != 0)
                std::cout << std::endl;

            std::cout << data.train_images[i * image_size + j] << " ";
        }

        std::cout << "\n\n";
    }

    for (auto i = 0; i < data.train_images.size; ++i)
        data.train_images[i] /= 255.0f;

    for (auto i = 0; i < data.test_images.size; ++i)
        data.test_images[i] /= 255.0f;

    data.train_images.reshape({60000, 28, 28, 1});
    data.test_images.reshape({10000, 28, 28, 1});

    data.train_labels = one_hot(data.train_labels, 10);
    data.test_labels = one_hot(data.test_labels, 10);

    cnn2d_train(data.train_images, data.train_labels);
    auto test_loss = cnn2d_evaluate(data.test_images, data.test_labels);
    cnn2d_predict(data.test_images, data.test_labels);

    return 0;
}