#include "arrs.h"
#include "rand.h"
#include "tensor.h"

std::vector<std::pair<size_t, size_t>> max_indices;

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

int main () {
    auto dl_ds4 = uniform_dist({2, 2, 3, 3}, 0.0f, 0.000001f);
    auto dl_dc3 = zeros({2, 2, 6, 6});
    auto c3 = uniform_dist({2, 2, 6, 6}, 0.0f, 0.000001f);
    auto s4 = lenet_max_pool(c3);

    std::cout << c3 << "\n";
    std::cout << s4 << "\n";
    std::cout << dl_ds4 << "\n";

    size_t idx = 0;
    size_t cumulative_height = 0;
    size_t num_imgs = c3.shape.front() * c3.shape[1];
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

    std::cout << dl_dc3 << "\n";

    return 0;
}