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

tensor kernel1 = normal_dist({3, 3});
tensor kernel2 = normal_dist({3, 3});

tensor w1 = normal_dist({32 * 7 * 7});
tensor b1 = zeros({1, 1});

tensor w2 = normal_dist({128});
tensor b2 = zeros({1, 1});

tensor cnn2d_convolution(const tensor& x, const tensor& kernel, const size_t stride = 1, const size_t padding = 0) {
    // Add padding to the input matrix here? For example,
    //        0 0 0 0
    // 1 1 -> 0 1 1 0
    // 1 1    0 1 1 0
    //        0 0 0 0

    size_t kernel_height = 3, kernel_width = 3;
    // size_t input_height = 28, input_width = 28;
    size_t input_height = 9, input_width = 9;

    size_t output_height = (input_height - kernel_height) / stride + 1;
    size_t output_width = (input_width - kernel_width) / stride + 1;

    tensor output = zeros({output_height, output_width});

    for (int i = 0; i < output_height; ++i) {
        for (int j = 0; j < output_width; ++j) {
            float sum = 0.0;
            for (int m = 0; m < kernel_height; ++m) {
                for (int n = 0; n < kernel_width; ++n) {
                    sum += x(i + m, j + n) * kernel(m, n);
                }
            }
            output(i, j) = sum;
        }
    }

    return output;
}

tensor cnn2d_max_pool(const tensor& x) {

    return tensor();
}

tensor cnn2d_forward(const tensor& x) {
    auto x_conv1 = cnn2d_convolution(x, kernel1);
    x_conv1 = relu(x_conv1);
    x_conv1 = cnn2d_max_pool(x_conv1);

    auto x_conv2 = cnn2d_convolution(x_conv1, kernel2);
    x_conv2 = relu(x_conv2);
    x_conv2 = cnn2d_max_pool(x_conv2);

    auto x_fc = matmul(w1, x_conv2) + b1;
    x_fc = matmul(w2, x_fc) + b2;

    return x_fc;
}

void cnn2d_train(const tensor& x_train, const tensor& y_train) {
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

float cnn2d_evaluate(const tensor& x_test, const tensor& y_test) {
    return 0.0f;
}

void cnn2d_predict(const tensor& x_test, const tensor& y_test) {
}

int main() {
    // mnist data = load_mnist();

    // constexpr size_t num_digits = 10;
    // constexpr size_t image_size = 784;
    // constexpr size_t image_dim = 28;

    // for (auto i = 0; i < num_digits; ++i) {
    //     for (auto j = 0; j < image_size; ++j) {
    //         if (j % image_dim == 0 && j != 0)
    //             std::cout << std::endl;

    //         std::cout << data.train_images[i * image_size + j] << " ";
    //     }

    //     std::cout << "\n\n";
    // }

    // for (auto i = 0; i < data.train_images.size; ++i)
    //     data.train_images[i] /= 255.0f;

    // for (auto i = 0; i < data.test_images.size; ++i)
    //     data.test_images[i] /= 255.0f;

    // data.train_images.reshape({60000, 28, 28, 1});
    // data.test_images.reshape({10000, 28, 28, 1});

    // data.train_labels = one_hot(data.train_labels, 10);
    // data.test_labels = one_hot(data.test_labels, 10);

    // cnn2d_train(data.train_images, data.train_labels);
    // auto test_loss = cnn2d_evaluate(data.test_images, data.test_labels);
    // cnn2d_predict(data.test_images, data.test_labels);

    auto a = tensor({9, 9}, {0,  1,  2,  3,  4,  5,  6,  7,  8,
                             9, 10, 11, 12, 13, 14, 15, 16, 17,
                            18, 19, 20, 21, 22, 23, 24, 25, 26,
                            27, 28, 29, 30, 31, 32, 33, 34, 35,
                            36, 37, 38, 39, 40, 41, 42, 43, 44,
                            45, 46, 47, 48, 49, 50, 51, 52, 53,
                            54, 55, 56, 57, 58, 59, 60, 61, 62,
                            63, 64, 65, 66, 67, 68, 69, 70, 71,
                            72, 73, 74, 75, 76, 77, 78, 79, 80, });
    auto kernel_3x3 = tensor({3, 3}, {1});

    auto output = cnn2d_convolution(a, kernel_3x3);

    std::cout << a << std::endl;
    std::cout << output << std::endl;

    return 0;
}