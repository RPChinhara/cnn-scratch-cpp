#include "arrs.h"
#include "rand.h"
#include "tensor.h"

#include <chrono>

tensor lenet_convolution(const tensor& x, const tensor& kernels, const size_t stride = 1) {
    size_t num_kernels = kernels.shape.front();
    size_t kernel_height = kernels.shape[kernels.shape.size() - 2];
    size_t kernel_width = kernels.shape.back();

    size_t input_height = x.shape[x.shape.size() - 2];
    size_t input_width = x.shape.back();

    size_t output_height = (input_height - kernel_height) / stride + 1;
    size_t output_width = (input_width - kernel_width) / stride + 1;

    tensor outputs = zeros({x.shape.front(), num_kernels, output_height, output_width});

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
                    outputs[idx * output.size + i] = output[i];

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
                    outputs[idx * channels_sum.size + i] = channels_sum[i];

                ++idx;
            }
        }
    }

    return outputs;
}

int main() {
    tensor x1 = uniform_dist({2, 4, 4}, 0.0f, 0.0000001f);
    tensor x2 = uniform_dist({2, 2, 3, 3}, 0.0f, 0.0000001f);

    tensor kernel1 = zeros({1, 2, 2});
    for (size_t i = 0; i < kernel1.size; ++i) {
        kernel1[i] += 1.0f;
    }

    tensor kernel2 = zeros({2, 2, 2});
    for (size_t i = 0; i < kernel2.size; ++i) {
        if (i < 4)
            kernel2[i] += 1.0f;
        else
            kernel2[i] += 2.0f;
    }

    std::cout << x2 << "\n";
    std::cout << kernel1 << "\n";

    auto start = std::chrono::high_resolution_clock::now();

    // std::cout << lenet_convolution(x1, kernel1) << "\n";
    std::cout << lenet_convolution(x2, kernel2) << "\n";

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    std::cout << std::endl << "Time taken: " << duration.count() << " seconds\n\n";

    // (60000, 32, 32)
    // (60000, 6, 28, 28)
    // (60000, 6, 14, 14)

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
    // What is dot product between (2 x 2 x 2) and (2 x 2 x 2) of elements all 1?

    // just for loop i, j, and k ->
    // for(size_t i = 0; i < 1; ++i)
    //     for(size_t j = 0; j < 1; ++j)
    //         for(size_t k = 0; k < 1; ++k)
    // |2 2  2 2|   |2 2  2 2|
    // |2 2  2 2| x |2 2  2 2|

    return 0;
}