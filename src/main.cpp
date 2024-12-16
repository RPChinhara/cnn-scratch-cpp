#include "acts.h"
#include "arrs.h"
#include "datas.h"
#include "linalg.h"
#include "losses.h"
#include "math.h"
#include "rand.h"
#include "tensor.h"

#include <chrono>

constexpr size_t input_size = 400;
constexpr size_t hidden1_size = 120;
constexpr size_t hidden2_size = 84;
constexpr size_t output_size = 10;

tensor kernel1 = glorot_uniform({6, 5, 5});
tensor kernel2 = glorot_uniform({16, 5, 5});

tensor w1 = glorot_uniform({hidden1_size, input_size});
tensor w2 = glorot_uniform({hidden2_size, hidden1_size});
tensor w3 = glorot_uniform({output_size, hidden2_size});

tensor b1 = zeros({hidden1_size, 1});
tensor b2 = zeros({hidden2_size, 1});
tensor b3 = zeros({output_size, 1});

void print_imgs(const tensor& imgs, size_t num_digits) {
    size_t img_size = imgs.shape[1] * imgs.shape.back();
    size_t img_dim  = imgs.shape[1];

    for (auto i = 0; i < num_digits; ++i) {
        for (auto j = 0; j < img_size; ++j) {
            if (j % img_dim == 0 && j != 0)
                std::cout << std::endl;

            std::cout << imgs[i * img_size + j] << " ";
        }
        std::cout << "\n\n";
    }
}

tensor lenet_convolution(const tensor& x, const tensor& kernels, const size_t stride = 1) {
    size_t num_kernels = kernels.shape.front();
    size_t kernel_height = kernels.shape[kernels.shape.size() - 2];
    size_t kernel_width = kernels.shape.back();

    size_t input_height = x.shape[x.shape.size() - 2];
    size_t input_width = x.shape.back();

    size_t output_height = (input_height - kernel_height) / stride + 1;
    size_t output_width = (input_width - kernel_width) / stride + 1;

    tensor feature_maps = zeros({x.shape.front(), num_kernels, output_height, output_width});

    size_t idx = 0;

    if (x.shape.size() == 3) {
        size_t num_img = x.shape.front();

        for (size_t b = 0; b < num_img; ++b) {
            tensor img = slice(x, b * input_height, input_height);
            tensor output = zeros({output_height, output_width});

            for (size_t k = 0; k < num_kernels; ++k) {
                tensor kernel = slice(kernels, k * kernel_height, kernel_height);

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
                    feature_maps[idx * output.size + i] = output[i];

                ++idx;
            }
        }
    } else if (x.shape.size() == 4) {
        size_t num_batches = x.shape.front();
        size_t num_channels = x.shape[1];

        // TODO: Can I only use for loops twice like above when the x.shape.size() == 3?
        for (size_t i = 0; i < num_batches; ++i) {
            for (size_t j = 0; j < num_kernels; ++j) {
                tensor kernel = slice(kernels, j * kernel_height, kernel_height);
                tensor channels_sum = zeros({output_height, output_width});

                for (size_t k = 0; k < num_channels; ++k) {
                    size_t idx = i * num_channels + k;
                    tensor img = slice(x, idx * input_height, input_height);

                    // TODO: This could be optimized.
                    // If the size of kernel was 2 x 2, and img was 3 x 3.
                    // kernel, img
                    //         0 1 2
                    // 1 1     3 4 5
                    // 1 1     6 7 8
                    // (0) * (1), (1) * (1), (3) * (1), (4) * (1) -> this is dot product between top left corners.
                    // (1) * (1), (2) * (1), (4) * (1), (5) * (1) -> this is dot product between top right corners.
                    // This way I don't need to use 4 for loops like now.

                    for (size_t row = 0; row < output_height; ++row) {
                        for (size_t col = 0; col < output_width; ++col) {
                            float sum = 0.0;

                            for (size_t m = 0; m < kernel_height; ++m) {
                                for (size_t n = 0; n < kernel_width; ++n) {
                                    sum += img(row + m, col + n) * kernel(m, n);
                                }
                            }

                            channels_sum(row, col) += sum;
                        }
                    }
                }

                for (size_t i = 0; i < channels_sum.size; ++i)
                    feature_maps[idx * channels_sum.size + i] = channels_sum[i];

                ++idx;
            }
        }
    }

    return feature_maps;
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
    tensor c1 = relu(lenet_convolution(x, kernel1));
    tensor s2 = lenet_max_pool(c1);
    tensor c3 = relu(lenet_convolution(s2, kernel2));
    tensor s4 = lenet_max_pool(c3);

    s4.reshape({60000, 400});

    tensor f5 = relu(matmul(w1, transpose(s4)) + b1);
    tensor f6 = relu(matmul(w2, f5) + b2);
    tensor y = softmax(matmul(w3, f6) + b3);

    std::cout << x.get_shape() << "\n";
    std::cout << c1.get_shape() << "\n";
    std::cout << s2.get_shape() << "\n";
    std::cout << c3.get_shape() << "\n";
    std::cout << s4.get_shape() << "\n";
    std::cout << f5.get_shape() << "\n";
    std::cout << f6.get_shape() << "\n";
    std::cout << y.get_shape() << "\n";

    return y;
}

void lenet_train(const tensor& x_train, const tensor& y_train) {
    constexpr size_t epochs = 1;
    constexpr float  lr = 0.01f;
    constexpr size_t batch_size = 32;

    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        tensor y = lenet_forward(x_train);

        std::cout << y_train.get_shape() << "\n";

        float error = categorical_cross_entropy(y_train, transpose(y));

        // tensor d_loss_d_y = -2.0f / num_samples * (transpose(y_train) - y_sequence.front());

        tensor d_loss_d_kernel1 = zeros({6, 5, 5});
        tensor d_loss_d_kernel2 = zeros({16, 5, 5});

        tensor d_loss_d_w1 = zeros({hidden1_size, input_size});
        tensor d_loss_d_w2 = zeros({hidden2_size, hidden1_size});
        tensor d_loss_d_w3 = zeros({output_size, hidden2_size});

        tensor d_loss_b1 = zeros({hidden1_size, 1});
        tensor d_loss_b2 = zeros({hidden2_size, 1});
        tensor d_loss_b3 = zeros({output_size, 1});

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

        // kernel1 = kerne1 - lr * d_loss_d_kernel1;
        // kernel2 = kerne2 - lr * d_loss_d_kernel2;

        // w1 = w1 - lr * d_loss_d_w1;
        // w2 = w2 - lr * d_loss_d_w2;
        // w3 = w3 - lr * d_loss_d_w3;

        // b1 = b1 - lr * d_loss_b1;
        // b2 = b2 - lr * d_loss_b2;
        // b3 = b3 - lr * d_loss_b3;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << error << std::endl;
    }
}

float lenet_evaluate(const tensor& x_test, const tensor& y_test) {
    // auto y = lenet_forward(x_test);
    // return categorical_cross_entropy(y_test, transpose(y));

    return 0.0f;
}

void lenet_predict(const tensor& x_test, const tensor& y_test) {
}

int main() {
    mnist data = load_mnist();

    print_imgs(data.train_imgs, 1);

    data.train_imgs = pad(data.train_imgs, 2, 2, 2, 2);
    data.test_imgs = pad(data.test_imgs, 2, 2, 2, 2);

    print_imgs(data.train_imgs, 1);

    for (auto i = 0; i < data.train_imgs.size; ++i)
        data.train_imgs[i] /= 255.0f;

    for (auto i = 0; i < data.test_imgs.size; ++i)
        data.test_imgs[i] /= 255.0f;

    data.train_labels = one_hot(data.train_labels, 10);
    data.test_labels = one_hot(data.test_labels, 10);

    auto start = std::chrono::high_resolution_clock::now();

    lenet_train(data.train_imgs, data.train_labels);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << std::endl << "Time taken: " << duration.count() << " seconds\n";

    auto test_loss = lenet_evaluate(data.test_imgs, data.test_labels);
    std::cout << "Test loss:  " << test_loss << "\n\n";

    lenet_predict(data.test_imgs, data.test_labels);

    return 0;
}