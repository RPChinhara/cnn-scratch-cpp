#include "acts.h"
#include "arrs.h"
#include "datas.h"
#include "linalg.h"
#include "losses.h"
#include "math.h"
#include "rand.h"
#include "tensor.h"

#include <array>
#include <chrono>

tensor kernel1 = glorot_uniform({6, 5, 5});
tensor kernel2 = glorot_uniform({16, 5, 5});

tensor w1 = glorot_uniform({120, 400});
tensor w2 = glorot_uniform({84, 120});
tensor w3 = glorot_uniform({10, 84});

tensor b1 = zeros({120, 1});
tensor b2 = zeros({84, 1});
tensor b3 = zeros({10, 1});

// TODO: I have to have this for both s2 and s4
// TODO: I have to call clear() somewhere for proper indices for gradients calculation.
std::vector<std::pair<size_t, size_t>> max_indices;

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

// TODO: Move this to the lyrs folders? Unlike rnn, gru, lstm, conv2d and max_pool2d will be all same? Also, I could make max_pool2d_derivative in the file as well.
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

// TODO: Move this to the lyrs folders?
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
                std::pair<size_t, size_t> max_idx;

                for (size_t m = 0; m < pool_size; ++m) {
                    for (size_t n = 0; n < pool_size; ++n) {
                        float val = img(i * stride + m, j * stride + n);

                        if (val > max_val) {
                            max_idx.first = i * stride + m;
                            max_idx.second = j * stride + n;
                            max_val = val;
                        }
                    }
                }

                output(i, j) = max_val;
                max_indices.push_back(max_idx);
            }
        }

        for (size_t i = 0; i < output.size; ++i)
            outputs[b * output.size + i] = output[i];
    }

    return outputs;
}

std::array<tensor, 9> lenet_forward(const tensor& x) {
    // TODO: I have to compute gradients dc1/dc1_z as below, same for other that use activation funcs
    // tensor c1_z = lenet_convolution(x, kernel1);
    // tensor c1 = relu(c1_z);

    // NOTE: Do I need to biases for c1 to s4?

    tensor c1_z = lenet_convolution(x, kernel1);
    tensor c1 = relu(c1_z);

    tensor s2 = lenet_max_pool(c1);

    tensor c3_z = lenet_convolution(s2, kernel2);
    tensor c3 = relu(c3_z);

    tensor s4 = lenet_max_pool(c3);

    s4.reshape({60000, 400});

    tensor f5 = relu(matmul(w1, transpose(s4)) + b1);
    tensor f6 = relu(matmul(w2, f5) + b2);
    tensor y = softmax(matmul(w3, f6) + b3);

    std::array<tensor, 9> outputs;

    outputs[0] = c1_z;
    outputs[1] = c1;
    outputs[2] = s2;
    outputs[3] = c3_z;
    outputs[4] = c3;
    outputs[5] = s4;
    outputs[6] = f5;
    outputs[7] = f6;
    outputs[8] = y;

    return outputs;
}

void lenet_train(const tensor& x_train, const tensor& y_train) {
    constexpr size_t epochs = 10;
    constexpr size_t batch_size = 32;
    constexpr float lr = 0.01f;

    for (size_t i = 1; i <= epochs; ++i) {
        std::cout << "Epoch " << i << "/" << epochs << std::endl;

        auto start_time = std::chrono::high_resolution_clock::now();

        auto [c1_z, c1, s2, c3_z, c3, s4, f5, f6, y] = lenet_forward(x_train);

        float error = categorical_cross_entropy(y_train, transpose(y));

        tensor dl_dy = y - transpose(y_train);
        tensor dl_df6 = matmul(transpose(w3), dl_dy); // (84, 10), (10, 60000) = (84, 60000)
        tensor dl_df5 = matmul(transpose(w2), dl_df6); // (120, 60000)
        tensor dl_ds4 = matmul(transpose(w1), dl_df5).reshape({60000, 16, 5, 5});

        tensor dl_dc3 = zeros({60000, 16, 10, 10});

        // c3 from lenet_forward(): (60000, 16, 10, 10)
        // s4 from lenet_forward(): (60000, 400)

        // TODO: Make max_unpool()?
        size_t idx = 0;
        size_t cumulative_height = 0;
        size_t num_imgs = c3.shape.front() * c3.shape[1]; // TODO: Use constexpr hardcoding for better perf?
        size_t output_img_size = dl_ds4.shape[2] * dl_ds4.shape.back();

        for (size_t i = 0; i < num_imgs; ++i) {
            size_t img_height = c3.shape[2];
            // auto img = slice(x2, i * img_height, img_height);

            for (size_t j = 0; j < output_img_size; ++j) {
                // TODO: Use eigther of these below
                // img(max_indices[idx].first, max_indices[idx].second) = 1.0f;

                // TODO: Write notes.txt that I omitted to assign 1.0f, and directly assigned dl_ds4
                // dl_dc3(cumulative_height + max_indices[idx].first, max_indices[idx].second) = 1.0f;
                dl_dc3(cumulative_height + max_indices[idx].first, max_indices[idx].second) = dl_ds4[idx];

                ++idx;
            }

            cumulative_height += img_height;
        }

        tensor dl_dc3_z = dl_dc3 * relu_derivative(c3_z);

        std::cout << dl_dc3_z.get_shape() << "\n";

        tensor dl_dkernel2 = zeros({16, 5, 5});

        for (size_t i = 0; i < 3; ++i) {
            auto img = slice_4d(s2, i * s2.shape[1] * s2.shape[2] * s2.shape.back());
            auto kernel = slice_4d(dl_dc3_z, i * dl_dc3_z.shape[1] * dl_dc3_z.shape[2] * dl_dc3_z.shape.back());
            kernel.reshape({16, 10, 10});

            dl_dkernel2 += lenet_convolution(img, kernel).reshape({16, 5, 5});
        }

        std::cout << dl_dkernel2.get_shape() << "\n";

        tensor dl_ds2 = zeros({60000, 6, 14, 14});
        tensor dl_dc1 = zeros({60000, 6, 28, 28});

        tensor dl_dw3 = matmul(dl_dy, transpose(f6));
        tensor dl_dw2 = matmul(dl_df6, transpose(f5));
        tensor dl_dw1 = matmul(dl_df5, s4);

        tensor dl_dkernel1;

        tensor dl_db3 = sum(dl_dy, 1);
        tensor dl_db2 = sum(dl_df6, 1);
        tensor dl_db1 = sum(dl_df5, 1);

        // kernel1 = kernel1 - lr * dl_dkernel1;
        kernel2 = kernel2 - lr * dl_dkernel2;

        w1 = w1 - lr * dl_dw1;
        w2 = w2 - lr * dl_dw2;
        w3 = w3 - lr * dl_dw3;

        b1 = b1 - lr * dl_db1;
        b2 = b2 - lr * dl_db2;
        b3 = b3 - lr * dl_db3;

        // dl_dkernel1 = dl_dy * dy_df6 * df6_df5 * df5_ds4 * ds4_dc3 * dc3_ds2 * ds2_dc1 * dc1_dkernel1
        // dl_dkernel2 = dl_dy * dy_df6 * df6_df5 * df5_ds4 * ds4_dc3 * dc3_dkernel2

        // dl_dw1 = dl_dy * dy_df6 * df6_df5 * df5_dw1
        // dl_dw2 = dl_dy * dy_df6 * df6_dw2
        // dl_dw3 = dl_dy * dy_dw3

        // dl_db1 = dl_dy * dy_df6 * df6_df5 * df5_db1
        // dl_db2 = dl_dy * dy_df6 * df6_db2
        // dl_db3 = dl_dy * dy_db3

        // x:  (60000, 32, 32)
        // c1: (60000, 6, 28, 28)
        // s2: (60000, 6, 14, 14)
        // c3: (60000, 16, 10, 10)
        // s4: before reshape -> (60000, 16, 5, 5), after reshape -> (60000, 400)
        // f5: (120, 60000)
        // f6: (84, 60000)
        // y:  (10, 60000)

        // w1: (120, 400)
        // w2: (84, 120)
        // w3: (10, 84)

        // b1: (120, 1)
        // b2: (84, 1)
        // b3: (10, 1)

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "1/1 [==============================] - " << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << error << std::endl;
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