#include "arrs.h"
#include "rand.h"
#include "tensor.h"

// TODO: Move this to the lyrs folders? Unlike rnn, gru, lstm, conv2d and max_pool2d will be all same? Also, I could make max_pool2d_derivative in the file as well.
tensor convolution(const tensor& x, const tensor& kernels, const size_t stride = 1) {
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

int main () {
                              // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                              // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                              // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                              // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
                              // 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0
                              // 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0
                              // 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0
                              // 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0
    // 1 1 1 1 1 1 1 1 1 1    // 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0
    // 1 1 1 1 1 1 1 1 1 1    // 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0
    // 1 1 1 1 1 1 1 1 1 1    // 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0
    // 1 1 1 1 1 1 1 1 1 1    // 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0
    // 1 1 1 1 1 1 1 1 1 1    // 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0
    // 1 1 1 1 1 1 1 1 1 1    // 0 0 0 0 1 1 1 1 1 1 1 1 1 1 0 0 0 0
    // 1 1 1 1 1 1 1 1 1 1    // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    // 1 1 1 1 1 1 1 1 1 1    // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    // 1 1 1 1 1 1 1 1 1 1    // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
    // 1 1 1 1 1 1 1 1 1 1 -> // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0



    tensor dl_dc3_z = uniform_dist({3, 16, 10, 10}, 0.0f, 0.0001f);
    tensor kernel2 = zeros({16, 5, 5});

    size_t padding_size = kernel2.shape[1] - 1;

    tensor dl_dc3_z_padded = pad(dl_dc3_z, padding_size, padding_size, padding_size, padding_size);


    for (size_t i = 0; i < 3; ++i) {
        auto img = slice_4d(s2, i);
        auto kernel = slice_4d(dl_dc3, i);
        kernel.reshape({16, 10, 10});

        dl_dkernel2 += convolution(img, kernel);

        std::cout << kernel << "\n";
        std::cout << img << "\n";
        std::cout << dl_dkernel2 << "\n";

    }

    std::cout << dl_dkernel2 << "\n";

    return 0;
}