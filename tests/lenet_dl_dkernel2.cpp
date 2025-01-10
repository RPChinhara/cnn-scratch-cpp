#include "arrs.h"
#include "rand.h"
#include "tensor.h"

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

    // Optimization: Can I only use for loops twice similar to when I had 'if (x.shape.size() == 3)'?
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

                // Optimization: This could be optimized.
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

int main () {
    size_t batch_size = 2;

    // tensor s2 = uniform_dist({batch_size, 6, 14, 14}, 0.0f, 0.0001f); // real shape = (64, 6, 14, 14)
    // tensor dl_dc3 = uniform_dist({batch_size, 16, 10, 10}, 0.0f, 0.0001f); // real shape = (64, 16, 10, 10)
    // tensor dl_dkernel2 = zeros({16, 6, 5, 5});

    // TODO: test if it does work after supporting batch size
    // NOTE: Smaller shapes for test
    tensor s2 = zeros({batch_size, 3, 4, 4});
    tensor dl_dc3 = zeros({batch_size, 2, 2, 2});
    tensor dl_dkernel2 = zeros({2, 3, 3, 3});

    for (size_t i = 0; i < s2.size; ++i)
        s2[i] = i;

    for (size_t i = 0; i < dl_dc3.size; ++i)
        dl_dc3[i] = i;

    size_t input_channels = s2.shape[1];
    size_t output_channels = dl_dc3.shape[1];

    std::cout << s2 << std::endl;
    std::cout << dl_dc3 << std::endl;

    for (size_t i = 0; i < batch_size; ++i) {
        tensor s2_sample = slice_4d(s2, i, 1);
        tensor dl_dc3_sample = slice_4d(dl_dc3, i, 1);

        tensor dl_dkernel2_partial = zeros({16, 6, 5, 5});
        size_t idx = 0;

        for (size_t j = 0; j < output_channels; ++j) {
            // tensor dl_dc3_feature_map = slice(dl_dc3_sample, j * 10, 10);
            // dl_dc3_feature_map.reshape({1, 1, 10, 10});

            // NOTE: Smaller shapes for test
            tensor dl_dc3_feature_map = slice(dl_dc3_sample, j * 2, 2);
            dl_dc3_feature_map.reshape({1, 1, 2, 2});

            std::cout << dl_dc3_feature_map << std::endl;

            for (size_t k = 0; k < input_channels; ++k) {
                // tensor s2_feature_map = slice(s2_sample, k * 14, 14);
                // s2_feature_map.reshape({1, 1, 14, 14});

                // NOTE: Smaller shapes for test
                tensor s2_feature_map = slice(s2_sample, k * 4, 4);
                s2_feature_map.reshape({1, 1, 4, 4});

                std::cout << s2_feature_map << std::endl;

                tensor dl_dkernel2_feature_map = convolution(s2_feature_map, dl_dc3_feature_map);

                for (size_t l = 0; l < dl_dkernel2_feature_map.size; ++l)
                    dl_dkernel2_partial[idx * dl_dkernel2_feature_map.size + l] = dl_dkernel2_feature_map[l];

                ++idx;
            }
        }

        dl_dkernel2 += dl_dkernel2_partial;
    }

    std::cout << dl_dkernel2 << std::endl;

    return 0;
}