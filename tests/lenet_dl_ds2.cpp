#include "arrs.h"
#include "linalg.h"
#include "rand.h"
#include "tensor.h"

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

int main () {
     // NOTE: dl_ds2 = convolution(dl_dc3_z, kernel2); dl_dc3_z is padded to (batch_size, 16, 18, 18), and kernel2 is transposed to (6, 16, 5, 5). Don't forget to transpose the spatial dimensions (the 5s) as well!

    size_t batch_size = 32;

    tensor dl_dc3_z = zeros({batch_size, 16, 10, 10});
    for (size_t i = 0; i < dl_dc3_z.size; ++i) dl_dc3_z[i] = i;

    tensor kernel2 = zeros({16, 6, 5, 5});
    for (size_t i = 0; i < kernel2.size; ++i) kernel2[i] = i;

    tensor dl_ds2_test = deconvolution(dl_dc3_z, kernel2);

    std::cout << dl_ds2_test.get_shape() << std::endl;

    return 0;
}