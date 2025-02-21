#include "acts.h"
#include "arrs.h"
#include "datasets.h"
#include "linalg.h"
#include "losses.h"
#include "math.h"
#include "rand.h"
#include "tensor.h"

#include <array>
#include <chrono>

// NOTE: This and other files were supposed to be named lenet.cpp or lenet5.cpp, but now I consider all CNNs are same that they only differs by number of convolution layers and pooling, and some new techniques like skip connections in ResNet or the fact VGGNet uses small 3 x 3 filters and so on.

// NOTE: I could just have a one file callled cnn.cpp by unifying all the files, but I guess it's fine for now for experiments. Also, I could refactor and do some clean ups by making Sequential class like the one in TF so that I could make a model with different hyperparameters without manually doing it right now which is pretty tedious. I could do some optimization stuff as well. There are bunch of stuff I could do to make it better, however in order for me to proceed forward, I guess I'd just leave it as it is for now, and come back later when I need to use CNNs in the future.

tensor kernel1 = glorot_uniform({2, 1, 5, 5});
tensor kernel2 = glorot_uniform({16, 6, 5, 5});

size_t num_neurons = kernel1.shape[0] * 14 * 14;

tensor w1 = glorot_uniform({120, num_neurons});
tensor w2 = glorot_uniform({84, 120});
tensor w3 = glorot_uniform({10, 84});

tensor b1 = zeros({120, 1});
tensor b2 = zeros({84, 1});
tensor b3 = zeros({10, 1});

std::vector<std::pair<size_t, size_t>> indices_c1, indices_c3;

void print_imgs(const tensor& imgs, size_t num_digits) {
    size_t img_height = imgs.shape[imgs.shape.size() - 2];
    size_t img_size = img_height * img_height;

    for (auto i = 0; i < num_digits; ++i) {
        for (auto j = 0; j < img_size; ++j) {
            if (j % img_height == 0 && j != 0)
                std::cout << "\n";

            std::cout << imgs[i * img_size + j] << " ";
        }
        std::cout << "\n\n";
    }
}

// TODO: Move this to the lyrs folders? Unlike rnn, gru, lstm, conv2d and max_pool2d will be all same? Also, I could make max_pool2d_derivative in the file as well.
tensor convolution(const tensor& x, const tensor& kernels, const size_t stride = 1) {
    size_t input_channels = x.shape[1];
    size_t output_channels = kernels.shape.front();

    size_t kernel_height = kernels.shape[2];
    size_t kernel_width = kernels.shape.back();

    size_t input_height = x.shape[2];
    size_t input_width = x.shape.back();

    size_t output_height = (input_height - kernel_height) / stride + 1;
    size_t output_width = (input_width - kernel_width) / stride + 1;

    size_t batch_size = x.shape.front();

    tensor feature_maps = zeros({batch_size, output_channels, output_height, output_width});

    size_t idx = 0;

    // OPTIMIZE: Can I only use for loops twice similar to when I had 'if (x.shape.size() == 3)'?
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t j = 0; j < output_channels; ++j) {
            // tensor kernel_4d = slice_4d(kernels, j, 1); NOTE: Method2
            tensor channels_sum = zeros({output_height, output_width});

            // std::cout << kernel_4d << std::endl;

            for (size_t k = 0; k < input_channels; ++k) {
                size_t idx = i * input_channels + k;
                size_t idx2 = j * input_channels + k;
                tensor img = slice(x, idx * input_height, input_height);
                tensor kernel = slice(kernels, idx2 * kernel_height, kernel_height); // NOTE: Comment this out for 'Method2'
                // tensor kernel_2d = slice(kernel_4d, k * kernel_height, kernel_height); NOTE: Method2

                /*
                 * OPTIMIZE: This could be optimized.
                 * If the size of kernel was 2 x 2, and img was 3 x 3.
                 * kernel,  img
                 *          0 1 2
                 * 1 1      3 4 5
                 * 1 1      6 7 8
                 * (0) * (1), (1) * (1), (3) * (1), (4) * (1) -> this is dot product between top left corners.
                 * (1) * (1), (2) * (1), (4) * (1), (5) * (1) -> this is dot product between top right corners.
                 * This way I don't need to use 4 for loops like now.
                 */

                // OPTIMIZE: Can I increase indexes at same like this? 	for (i = j = 0; j < old->count; i++, j++) {
                for (size_t row = 0; row < output_height; ++row) {
                    for (size_t col = 0; col < output_width; ++col) {
                        float sum = 0.0;

                        for (size_t m = 0; m < kernel_height; ++m) {
                            for (size_t n = 0; n < kernel_width; ++n) {
                                sum += img(row + m, col + n) * kernel(m, n); // NOTE: kernel_2d(m, n) for 'Method2'
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

    return feature_maps;
}

tensor deconvolution(const tensor& input, const tensor& kernels) {
    size_t kernel_size = kernels.shape[2];
    size_t padding_size = kernel_size - 1;

    tensor padded_input = pad(input, padding_size, padding_size, padding_size, padding_size);

    tensor transposed_kernels = zeros({kernels.shape[1], kernels.shape[0], kernels.shape[3], kernels.shape[2]});

    for (size_t n = 0; n < kernels.shape[0]; ++n) {
        for (size_t c = 0; c < kernels.shape[1]; ++c) {
            for (size_t h = 0; h < kernels.shape[2]; ++h) {
                for (size_t w = 0; w < kernels.shape[3]; ++w) {
                    float value = kernels.get({n, c, h, w});

                    transposed_kernels.set({c, n, w, h}, value);
                }
            }
        }
    }

    return convolution(padded_input, transposed_kernels);
}

// NOTE: What happens if there were two max values in the region, which one should it pick?
std::pair<tensor, std::vector<std::pair<size_t, size_t>>> max_pool(const tensor& x, const size_t pool_size = 2, const size_t stride = 2) {
    size_t input_channels = x.shape[1];

    size_t input_height = x.shape[2];
    size_t input_width = x.shape.back();

    size_t output_height = (input_height - pool_size) / stride + 1;
    size_t output_width = (input_width - pool_size) / stride + 1;

    size_t batch_size = x.shape.front();

    tensor outputs = zeros({batch_size, input_channels, output_height, output_width});

    size_t num_img = batch_size * input_channels;

    std::vector<std::pair<size_t, size_t>> max_indices;

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

    return {outputs, max_indices};
}

tensor max_unpool(const tensor& input,  const std::vector<std::pair<size_t, size_t>>& indices) {
    size_t input_height = input.shape[2];
    size_t input_width = input.shape.back();

    // NOTE: This is if following values are used in max pooling. Kernel size = 2, Stride = 2, and Padding = 0.
    size_t output_height = 2 * input_height;
    size_t output_width = 2 * input_width;

    size_t num_imgs = input.shape.front() * input.shape[1];
    size_t pooled_img_size = input.shape[2] * input.shape.back();

    tensor output = zeros({input.shape.front(), input.shape[1], output_height, output_width});

    size_t idx = 0;

    for (size_t i = 0; i < num_imgs; ++i) {
        for (size_t j = 0; j < pooled_img_size; ++j) {
            output(i * output_height + indices[idx].first, indices[idx].second) = input[idx];
            ++idx;
        }
    }

    return output;
}

std::array<tensor, 7> forward(const tensor& x, float batch_size) {
    // NOTE: Do I need biases convolution?
    // NOTE: Do I need to make sequential model like tensorflow does? I realised it was fine when working with sequential models like lstm, gru, but for models like nn and cnn, might be useful?

    indices_c1.clear();
    indices_c3.clear();

    tensor c1_z = convolution(x, kernel1);
    tensor c1 = sigmoid(c1_z);

    auto [s2, indices_c1_temp] = max_pool(c1);
    indices_c1 = indices_c1_temp;

    s2.reshape({static_cast<size_t>(batch_size), num_neurons});

    tensor f5_z = matmul(w1, transpose(s2)) + b1; // w1: (120, num_neurons)
    tensor f5 = sigmoid(f5_z);

    tensor f6_z = matmul(w2, f5) + b2; // w2: (84, 120)
    tensor f6 = sigmoid(f6_z);

    // NOTE: Inputs have to be transposed so that dimension muchs with how softmax is created.
    tensor y = softmax(transpose(matmul(w3, f6) + b3)); // w3: (10, 84)

    std::array<tensor, 7> outputs;

    outputs[0] = c1_z;
    outputs[1] = s2;
    outputs[2] = f5_z;
    outputs[3] = f5;
    outputs[4] = f6_z;
    outputs[5] = f6;
    outputs[6] = y;

    return outputs;
}

void train(const tensor& x_train, const tensor& y_train) {
    constexpr size_t epochs = 10;
    constexpr float lr = 0.01f;
    float batch_size = 64.0f;

    const size_t num_batches = static_cast<size_t>(ceil(60000.0f / batch_size));

    for (size_t i = 1; i <= epochs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Epoch " << i << "/" << epochs << "\n";

        float loss = 0.0f;

        batch_size = 64.0f;

        // TODO: I have to process multiple batches simultaneously in order to speed up training lol That is why batach training is faster right?
        for (size_t j = 0; j < num_batches; ++j) {
            size_t start_idx = j * batch_size; // 937 x 64 = 59968
            size_t end_idx = std::min(start_idx + batch_size, 60000.0f);

            tensor x_batch = slice_4d(x_train, start_idx, end_idx - start_idx);
            tensor y_batch = slice(y_train, start_idx, end_idx - start_idx);

            if (j == num_batches - 1)
                batch_size = static_cast<float>(end_idx - start_idx);

            auto [c1_z, s2, f5_z, f5, f6_z, f6, y] = forward(x_batch, batch_size);

            std::cout << "    ";

            for (size_t s = 0; s < 10; ++s)
                std::cout << y_batch[s] << " ";

            std::cout << "    ";

            for (size_t s = 0; s < 10; ++s)
                std::cout << y[s] << " ";

            loss = categorical_cross_entropy(y_batch, y);

            tensor dl_dy = transpose(y - y_batch);
            tensor dl_df6 = matmul(transpose(w3), dl_dy); // (84, 10), (10, batch_size) = (84, batch_size)
            tensor dl_df6_z = dl_df6 * sigmoid_derivative(f6_z);
            tensor dl_df5 = matmul(transpose(w2), dl_df6_z); // (120, batch_size)
            tensor dl_df5_z = dl_df5 * sigmoid_derivative(f5_z);
            tensor dl_ds2 = matmul(transpose(w1), dl_df5_z).reshape({static_cast<size_t>(batch_size), kernel1.shape[0], 14, 14});

            tensor dl_dc1 = max_unpool(dl_ds2, indices_c1);
            tensor dl_dc1_z = dl_dc1 * sigmoid_derivative(c1_z);

            // tensor dl_dkernel2 = zeros({kernel2.shape[0], kernel2.shape[1], kernel2.shape[2], kernel2.shape[3]});
            tensor dl_dkernel1 = zeros({kernel1.shape[0], kernel1.shape[1], kernel1.shape[2], kernel1.shape[3]});

            tensor dl_dw3 = matmul(dl_dy, transpose(f6));
            tensor dl_dw2 = matmul(dl_df6_z, transpose(f5));
            tensor dl_dw1 = matmul(dl_df5_z, s2);

            tensor dl_db3 = sum(dl_dy, 1);
            tensor dl_db2 = sum(dl_df6_z, 1);
            tensor dl_db1 = sum(dl_df5_z, 1);

            // TODO: Make this a function for readability? I think it'd be useful for the future as well!
            // for (size_t i = 0; i < batch_size; ++i) {
            //     tensor s2_sample = slice_4d(s2, i, 1);
            //     tensor dl_dc3_sample = slice_4d(dl_dc3_z, i, 1);

            //     tensor dl_dkernel2_partial = zeros({16, 6, 5, 5});
            //     size_t idx = 0;

            //     for (size_t j = 0; j < 16; ++j) {
            //         tensor dl_dc3_feature_map = slice(dl_dc3_sample, j * 10, 10);
            //         dl_dc3_feature_map.reshape({1, 1, 10, 10});

            //         for (size_t k = 0; k < 6; ++k) {
            //             tensor s2_feature_map = slice(s2_sample, k * 14, 14);
            //             s2_feature_map.reshape({1, 1, 14, 14});

            //             tensor dl_dkernel2_feature_map = convolution(s2_feature_map, dl_dc3_feature_map);

            //             for (size_t l = 0; l < dl_dkernel2_feature_map.size; ++l)
            //                 dl_dkernel2_partial[idx * dl_dkernel2_feature_map.size + l] = dl_dkernel2_feature_map[l];

            //             ++idx;
            //         }
            //     }

            //     dl_dkernel2 += dl_dkernel2_partial;
            // }

            for (size_t i = 0; i < batch_size; ++i) {
                tensor x_sample = slice_4d(x_batch, i, 1);
                tensor dl_dc1_z_sample = slice_4d(dl_dc1_z, i, 1);

                tensor dl_dkernel1_partial = zeros({kernel1.shape[0], kernel1.shape[1], kernel1.shape[2], kernel1.shape[3]});
                size_t idx = 0;

                for (size_t j = 0; j < kernel1.shape[0]; ++j) {
                    tensor dl_dc1_z_feature_map = slice(dl_dc1_z_sample, j * 28, 28);
                    dl_dc1_z_feature_map.reshape({1, 1, 28, 28});

                    for (size_t k = 0; k < kernel1.shape[1]; ++k) {
                        tensor x_feature_map = slice(x_sample, k * 32, 32);
                        x_feature_map.reshape({1, 1, 32, 32});

                        tensor dl_dkernel1_feature_map = convolution(x_feature_map, dl_dc1_z_feature_map);

                        for (size_t l = 0; l < dl_dkernel1_feature_map.size; ++l)
                            dl_dkernel1_partial[idx * dl_dkernel1_feature_map.size + l] = dl_dkernel1_feature_map[l];

                        ++idx;
                    }
                }

                dl_dkernel1 += dl_dkernel1_partial;
            }

            kernel1 = kernel1 - lr * dl_dkernel1;
            // kernel2 = kernel2 - lr * dl_dkernel2;

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

            // x:  (batch_size, 1, 32, 32)
            // c1: (batch_size, 1, 28, 28)
            // s2: before reshape -> (batch_size, 1, 14, 14), after reshape -> (batch_size, 196)
            // f5: (120, batch_size)
            // f6: (84, batch_size)
            // y:  (10, batch_size)

            // w1: (120, 400)
            // w2: (84, 120)
            // w3: (10, 84)

            // b1: (120, 1)
            // b2: (84, 1)
            // b3: (10, 1)

            if (j == num_batches - 1)
                std::cout << "\r\033[K";
            else
                std::cout << "\r\033[K" << j + 1 << "/" << num_batches << " - loss: " << loss << std::flush;
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << num_batches << "/" << num_batches << " - " << std::chrono::duration<double>(end - start).count() << "s/step - loss: " << loss << "\n";
    }
}

float evaluate(const tensor& x_test, const tensor& y_test) {
    // auto y = forward(x_test);
    // return categorical_cross_entropy(y_test, transpose(y));

    return 0.0f;
}

void predict(const tensor& x_test, const tensor& y_test) {
}

int main() {
    mnist data = load_mnist();

    print_imgs(data.train_imgs, 1);

    data.train_imgs.reshape({60000, 1, 28, 28});
    data.test_imgs.reshape({10000, 1, 28, 28});

    for (auto i = 0; i < data.train_imgs.size; ++i)
        data.train_imgs[i] /= 255.0f;

    for (auto i = 0; i < data.test_imgs.size; ++i)
        data.test_imgs[i] /= 255.0f;

    data.train_imgs = pad(data.train_imgs, 2, 2, 2, 2);
    data.test_imgs = pad(data.test_imgs, 2, 2, 2, 2);

    print_imgs(data.train_imgs, 2);

    data.train_labels = one_hot(data.train_labels, 10);
    data.test_labels = one_hot(data.test_labels, 10);

    auto start = std::chrono::high_resolution_clock::now();
    train(data.train_imgs, data.train_labels);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << " seconds\n";

    std::cout << "Test loss: " << evaluate(data.test_imgs, data.test_labels) << "\n\n";

    predict(data.test_imgs, data.test_labels);

    return 0;
}