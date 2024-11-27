#include "acts.h"
#include "arrs.h"
#include "datas.h"
#include "linalg.h"
#include "losses.h"
#include "math.hpp"
#include "rand.h"
#include "tensor.h"

#include <chrono>

constexpr float  lr           = 0.01f;
constexpr size_t batch_size   = 32;
constexpr size_t epochs       = 10;

constexpr size_t input_size   = 256;
constexpr size_t hidden1_size = 120;
constexpr size_t hidden2_size = 84;
constexpr size_t output_size  = 10;

tensor kernel1 = glorot_uniform({6, 5, 5});
tensor kernel2 = glorot_uniform({16, 5, 5});

tensor w1 = glorot_uniform({hidden1_size, input_size});
tensor w2 = glorot_uniform({hidden2_size, hidden1_size});
tensor w3 = glorot_uniform({output_size, hidden2_size});

tensor b1 = zeros({hidden1_size, 1});
tensor b2 = zeros({hidden2_size, 1});
tensor b3 = zeros({output_size, 1});

void print_imgs(const tensor& imgs, size_t num_digits) {
    constexpr size_t img_size = 784;
    constexpr size_t img_dim = 28;

    for (auto i = 0; i < num_digits; ++i) {
        for (auto j = 0; j < img_size; ++j) {
            if (j % img_dim == 0 && j != 0)
                std::cout << std::endl;

            std::cout << imgs[i * img_size + j] << " ";
        }
        std::cout << "\n\n";
    }
}

tensor lenet_convolution(const tensor& x, const tensor& kernels, const size_t stride = 1, const size_t padding = 0) {
    // Add padding to the input matrix here? For example,
    //        0 0 0 0
    // 1 1 -> 0 1 1 0
    // 1 1    0 1 1 0
    //        0 0 0 0

    size_t num_kernels = kernels.shape.front();
    size_t kernel_height = kernels.shape[kernels.shape.size() - 2];
    size_t kernel_width = kernels.shape.back();

    size_t input_height = x.shape[x.shape.size() - 2];
    size_t input_width = x.shape.back();

    size_t output_height = (input_height - kernel_height) / stride + 1;
    size_t output_width = (input_width - kernel_width) / stride + 1;

    tensor outputs = zeros({x.shape.front(), num_kernels, output_height, output_width});

    size_t num_img;

    if (x.shape.size() == 3) {
        num_img = x.shape.front();
    } else if (x.shape.size() == 4) {
        num_img = x.shape.front() * x.shape[1];
    }

    size_t idx = 0;
    for (size_t b = 0; b < num_img; ++b) {
        auto img = slice(x, b * input_height, input_height);

        tensor output = zeros({output_height, output_width});

        for (size_t k = 0; k < num_kernels; ++k) {
            auto kernel = slice(kernels, k * kernel_height, kernel_height);

            for (size_t i = 0; i < output_height; ++i) {
                for (size_t j = 0; j < output_width; ++j) {
                    float sum = 0.0;

                    for (size_t m = 0; m < kernel_height; ++m) {
                        for (size_t n = 0; n < kernel_width; ++n) {
                            sum += img(i + m, j + n) * kernel(m, n);
                        }
                    }

                    output(i, j) = sum;
                }
            }

            for (size_t i = 0; i < output.size; ++i)
                outputs[idx * output.size + i] = output[i];

            ++idx;
        }
    }

    return outputs;
}

tensor lenet_max_pool(const tensor& x, const size_t pool_size = 2, const size_t stride = 2) {
    size_t num_kernels = x.shape[1];

    size_t input_height = x.shape[x.shape.size() - 2];
    size_t input_width = x.shape.back();

    size_t output_height = (input_height - pool_size) / stride + 1;
    size_t output_width = (input_width - pool_size) / stride + 1;

    tensor outputs = zeros({x.shape.front(), num_kernels, output_height, output_width});

    size_t batch_size = x.shape.front();
    size_t num_img = batch_size * num_kernels;

    for (size_t b = 0; b < num_img; ++b) {
        auto img = slice(x, b * input_height, input_height);

        tensor output = zeros({output_height, output_width});

        for (size_t i = 0; i < output_height; ++i) {
            for (size_t j = 0; j < output_width; ++j) {
                float max_val = -std::numeric_limits<float>::infinity();

                for (size_t m = 0; m < pool_size; ++m) {
                    for (size_t n = 0; n < pool_size; ++n) {
                        float val = img(i * stride + m, j * stride + n);

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
    std::cout << x.get_shape() << "\n";
    tensor c1 = lenet_convolution(x, kernel1);
    c1 = relu(c1);
    std::cout << c1.get_shape() << "\n";

    tensor s2 = lenet_max_pool(c1);
    std::cout << s2.get_shape() << "\n";

    tensor c3 = lenet_convolution(s2, kernel2);
    c3 = relu(c3);
    std::cout << c3.get_shape() << "\n";

    tensor s4 = lenet_max_pool(c3);
    std::cout << s4.get_shape() << "\n";

    // TODO: Can I do x_conv2.reshape({25, 60000});?
    s4.reshape({60000, 256});

    tensor f5 = matmul(w1, transpose(s4)) + b1;
    std::cout << f5.get_shape() << "\n";

    tensor f6 = matmul(w2, f5) + b2;
    std::cout << f6.get_shape() << "\n";

    tensor y = softmax(matmul(w3, f6) + b3);
    std::cout << y.get_shape() << "\n";

    return y;
}

void lenet_train(const tensor& x_train, const tensor& y_train) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        tensor y = lenet_forward(x_train);

        float error = categorical_cross_entropy(y_train, transpose(y));

        // tensor d_loss_d_y = -2.0f / num_samples * (transpose(y_train) - y_sequence.front());

        // for (auto j = seq_length; j > 0; --j) {
        //     if (j == seq_length) {
        //         tensor d_y_d_h_10 = w_hy;
        //         d_loss_d_h_t = matmul(transpose(d_loss_d_y), d_y_d_h_10);
        //     } else {
        //         d_loss_d_h_t = matmul(d_loss_d_h_t * transpose(relu_derivative(z_sequence[j])), w_hh);
        //     }

        //     d_loss_d_w_xh = d_loss_d_w_xh + matmul((transpose(d_loss_d_h_t) * relu_derivative(z_sequence[j - 1])), x_sequence[j - 1]);
        //     d_loss_d_w_hh = d_loss_d_w_hh + matmul((transpose(d_loss_d_h_t) * relu_derivative(z_sequence[j - 1])), transpose(h_sequence[j - 1]));

        //     d_loss_d_b_h  = d_loss_d_b_h + sum(transpose(d_loss_d_h_t) * relu_derivative(z_sequence[j - 1]), 1);
        // }

        // w = w - lr * d_loss_d_w;
        // b = b - lr * d_loss_d_y;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << error << std::endl;
    }
}

float lenet_evaluate(const tensor& x_test, const tensor& y_test) {
    auto y = lenet_forward(x_test);
    return categorical_cross_entropy(y_test, transpose(y));
}

void lenet_predict(const tensor& x_test, const tensor& y_test) {
}

int main() {
    // mnist data = load_mnist();

    // constexpr size_t num_digits = 2;
    // print_imgs(data.train_imgs, num_digits);

    // for (auto i = 0; i < data.train_imgs.size; ++i)
    //     data.train_imgs[i] /= 255.0f;

    // for (auto i = 0; i < data.test_imgs.size; ++i)
    //     data.test_imgs[i] /= 255.0f;

    // data.train_labels = one_hot(data.train_labels, 10);
    // data.test_labels = one_hot(data.test_labels, 10);

    // lenet_train(data.train_imgs, data.train_labels);
    // auto test_loss = lenet_evaluate(data.test_imgs, data.test_labels);
    // lenet_predict(data.test_imgs, data.test_labels);

    // NOTE: 1 to 2 is done?

    // 1.(60000, 28, 28)
    // 2.(60000, 6, 24, 24)
    // 3.(60000, 6, 12, 12)
    // (60000, 16, 8, 8)
    // (60000, 16, 4, 4)
    // (120, 60000)
    // (84, 60000)
    // (10, 60000)

    tensor x1 = uniform_dist({1, 3, 3}, 0.0f, 0.0000001f);
    tensor x2 = uniform_dist({1, 2, 2, 2}, 0.0f, 0.0000001f);
    tensor x3 = uniform_dist({3, 3}, 0.0f, 0.0000001f);

    tensor kernel1 = zeros({2, 2, 2});
    for (size_t i = 0; i < kernel1.size; ++i) {
        if (i < 4)
            kernel1[i] += 1.0f;
        else
            kernel1[i] += 2.0f;
    }

    tensor kernel2 = zeros({3, 2, 2});
    for (size_t i = 0; i < kernel2.size; ++i) {
        if (i < 4)
            kernel2[i] += 1.0f;
        else if (3 < i && i < 8)
            kernel2[i] += 2.0f;
        else
            kernel2[i] += 3.0f;
    }

    std::cout << x1 << "\n";
    std::cout << kernel1 << "\n";

    std::cout << lenet_convolution(x1, kernel1) << "\n";

    auto padded_x = pad(kernel1, 2);

    return 0;
}

// Tensor(
// [[[[0.00000005 0.00000010]  -> (1)
//    [0.00000005 0.00000000]]

//   [[0.00000001 0.00000002]  -> (2)
//    [0.00000002 0.00000008]]]], shape=(1, 2, 2, 2))
// Tensor(
// [[[1.00000000 1.00000000]   -> (3)
//   [1.00000000 1.00000000]]

//  [[2.00000000 2.00000000]   -> (4)
//   [2.00000000 2.00000000]]

//  [[3.00000000 3.00000000]   -> (5)
//   [3.00000000 3.00000000]]], shape=(3, 2, 2))


// The shape of the result be (1, 3, 1, 1). How? First multiply 1, 2, 3, then, 1, 2, 4, and 1, 2, 5.
// The reason is that size of kernel is 2 x 2 x 2, the first 2 is 1 and 2D, but for the last it means 3D which came from channel dim of inputs.
// Which is 2 next to 1.