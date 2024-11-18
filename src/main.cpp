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

constexpr float lr          = 0.01f;
constexpr size_t batch_size = 32;
constexpr size_t epochs     = 1;

tensor kernel1 = normal_dist({3, 3});
tensor kernel2 = normal_dist({3, 3});

tensor w1 = normal_dist({32 * 7 * 7});
tensor b1 = zeros({1, 1});

tensor w2 = normal_dist({128});
tensor b2 = zeros({1, 1});

tensor lenet_convolution(const tensor& x, const tensor& kernel, const size_t stride = 1, const size_t padding = 0) {
    // Add padding to the input matrix here? For example,
    //        0 0 0 0
    // 1 1 -> 0 1 1 0
    // 1 1    0 1 1 0
    //        0 0 0 0

    size_t kernel_height = kernel.shape.front();
    size_t kernel_width = kernel.shape.back();

    size_t input_height = x.shape[x.shape.size() - 2];
    size_t input_width = x.shape.back();

    size_t output_height = (input_height - kernel_height) / stride + 1;
    size_t output_width = (input_width - kernel_width) / stride + 1;

    tensor outputs = zeros({x.shape.front(), output_height, output_width});

    for (size_t b = 0; b < x.shape.front(); ++b) {
        auto t = slice(x, b * input_height, input_height);

        tensor output = zeros({output_height, output_width});

        for (size_t i = 0; i < output_height; ++i) {
            for (size_t j = 0; j < output_width; ++j) {
                float sum = 0.0;

                for (size_t m = 0; m < kernel_height; ++m) {
                    for (size_t n = 0; n < kernel_width; ++n) {
                        sum += t(i + m, j + n) * kernel(m, n);
                    }
                }

                output(i, j) = sum;
            }
        }

        for (size_t i = 0; i < output.size; ++i)
            outputs[b * output.size + i] = output[i];
    }

    return outputs;
}

tensor lenet_max_pool(const tensor& x, const size_t pool_size = 2, const size_t stride = 2) {
    size_t input_height = x.shape[x.shape.size() - 2];
    size_t input_width = x.shape.back();

    size_t output_height = (input_height - pool_size) / stride + 1;
    size_t output_width = (input_width - pool_size) / stride + 1;

    tensor outputs = zeros({x.shape.front(), output_height, output_width});

    for (size_t b = 0; b < x.shape.front(); ++b) {
        auto t = slice(x, b * input_height, input_height);

        tensor output = zeros({output_height, output_width});

        for (size_t i = 0; i < output_height; ++i) {
            for (size_t j = 0; j < output_width; ++j) {
                float max_val = -std::numeric_limits<float>::infinity();

                for (size_t m = 0; m < pool_size; ++m) {
                    for (size_t n = 0; n < pool_size; ++n) {
                        float val = t(i * stride + m, j * stride + n);

                        if (val > max_val)
                            max_val = val;
                    }
                }

                output(i, j) = max_val;
            }
        }

        for (size_t i = 0; i < output.size; ++i)
            outputs[b * output.size + i] = output[i];
    }

    return outputs;
}

tensor lenet_forward(const tensor& x) {
    auto x_conv1 = lenet_convolution(x, kernel1);
    x_conv1 = relu(x_conv1);
    x_conv1 = lenet_max_pool(x_conv1);

    auto x_conv2 = lenet_convolution(x_conv1, kernel2);
    x_conv2 = relu(x_conv2);
    x_conv2 = lenet_max_pool(x_conv2);

    // auto x_fc = matmul(w1, x_conv2) + b1;
    // x_fc = matmul(w2, x_fc) + b2;

    // return x_fc;

    return tensor();
}

void lenet_train(const tensor& x_train, const tensor& y_train) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto y = lenet_forward(x_train);

        float error = 0.0f;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << error << std::endl;
    }
}

float lenet_evaluate(const tensor& x_test, const tensor& y_test) {
    return 0.0f;
}

void lenet_predict(const tensor& x_test, const tensor& y_test) {
}

int main() {
    mnist data = load_mnist();

    constexpr size_t num_digits = 1;
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

    data.train_labels = one_hot(data.train_labels, 10);
    data.test_labels = one_hot(data.test_labels, 10);

    lenet_train(data.train_images, data.train_labels);
    auto test_loss = lenet_evaluate(data.test_images, data.test_labels);
    lenet_predict(data.test_images, data.test_labels);

    return 0;
}