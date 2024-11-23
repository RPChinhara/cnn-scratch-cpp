#include "arrs.h"
#include "tensor.h"

tensor lenet_convolution(const tensor& x, const tensor& kernels, const size_t stride = 1, const size_t padding = 0) {
    if (kernels.shape.size() != 3) {
        std::cerr << __FILE__ << "(" << __LINE__ << "): error: size of kernel should be 3" << std::endl;
        exit(1);
    }

    // Add padding to the input matrix here? For example,
    //        0 0 0 0
    // 1 1 -> 0 1 1 0
    // 1 1    0 1 1 0
    //        0 0 0 0

    size_t num_kernels = kernels.shape.front();
    size_t kernel_height = kernels.shape[1];
    size_t kernel_width = kernels.shape.back();

    size_t input_height = x.shape[x.shape.size() - 2];
    size_t input_width = x.shape.back();

    size_t output_height = (input_height - kernel_height) / stride + 1;
    size_t output_width = (input_width - kernel_width) / stride + 1;

    tensor outputs = zeros({x.shape.front(), num_kernels, output_height, output_width});

    std::cout << num_kernels << std::endl;
    size_t idx = 0;
    for (size_t b = 0; b < x.shape.front(); ++b) {
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

// (60000, 28, 28)
// (60000, 6, 24, 24)
// (60000, 6, 12, 12)
// (60000, 16, 8, 8)
// (60000, 16, 4, 4)
// (120, 60000)
// (84, 60000)
// (10, 60000)

int main () {
    tensor x = tensor(
        {2, 2, 3, 3},
        {
           44,   2,  22,
            4,   8,   6,
            5,   4,  66,

            6,   5,   3,
            4,   7,   8,
            7,  32,   7,

            1, 288,   8,
            7,   4,   6,
            5,  32,   6,

           56,   1,  24,
            4,   4,   6,
           22,   5,   6
        }
    );

    tensor kernel = tensor({2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1});

    std::cout << lenet_convolution(x, kernel) << "\n";

    return 0;
}