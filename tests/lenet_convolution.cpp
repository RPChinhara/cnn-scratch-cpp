#include "arrs.h"
#include "rand.h"
#include "tensor.h"

#include <chrono>

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

int main() {
    tensor x = zeros({2, 2, 3, 3});
    for (size_t i = 0; i < x.size; ++i) {
        x[i] += i;
    }

    // NOTE: (output_channels, input_channels, kernel_size[0], kernel_size[1])
    // I guess this works as x and kernel input_channels much which are both 2 in this case.
    tensor kernel = zeros({3, 2, 2, 2});
    for (size_t i = 0; i < kernel.size; ++i) {
        if (i < 4)
            kernel[i] += 1.0f;
        else if (3 < i && i < 8)
            kernel[i] += 2.0f;
        else if (7 < i && i < 12)
            kernel[i] += 3.0f;
        else if (11 < i && i < 16)
            kernel[i] += 4.0f;
        else if (15 < i && i < 20)
            kernel[i] += 5.0f;
        else if (19 < i && i < 24)
            kernel[i] += 6.0f;
    }

    std::cout << x << "\n";
    std::cout << kernel << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    std::cout << convolution(x, kernel) << "\n";

    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Elapsed time: " << std::chrono::duration<double>(end - start).count() << " seconds\n";

    return 0;
}