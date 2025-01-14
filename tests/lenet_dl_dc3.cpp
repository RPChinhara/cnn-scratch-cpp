#include "arrs.h"
#include "rand.h"
#include "tensor.h"

std::vector<std::pair<size_t, size_t>> max_indices;

tensor max_pool(const tensor& x, const size_t pool_size = 2, const size_t stride = 2) {
    size_t input_channels = x.shape[1];

    size_t input_height = x.shape[2];
    size_t input_width = x.shape.back();

    size_t output_height = (input_height - pool_size) / stride + 1;
    size_t output_width = (input_width - pool_size) / stride + 1;

    size_t batch_size = x.shape.front();

    tensor outputs = zeros({batch_size, input_channels, output_height, output_width});

    size_t num_img = batch_size * input_channels;

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
    auto s4 = max_pool(c3);

    std::cout << c3 << "\n";
    std::cout << s4 << "\n";
    std::cout << dl_ds4 << "\n";

    size_t num_imgs = c3.shape.front() * c3.shape[1];
    size_t output_img_size = dl_ds4.shape[2] * dl_ds4.shape.back();
    size_t cumulative_height = 0;
    size_t idx = 0;
    size_t img_height = c3.shape[2];

    // TODO: Make MaxUnpool2d(), and pass input and the indices of the maximal values. This is in PyTorch. This way I could use this for dl_dc1, and also for AlexNEt, VGG, and ResNet.
    for (size_t i = 0; i < num_imgs; ++i) {
        for (size_t j = 0; j < output_img_size; ++j) {
            dl_dc3(cumulative_height + max_indices[idx].first, max_indices[idx].second) = dl_ds4[idx];

            ++idx;
        }

        cumulative_height += img_height;
    }

    std::cout << dl_dc3 << "\n";

    return 0;
}