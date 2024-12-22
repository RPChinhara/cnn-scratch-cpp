#include "acts.h"
#include "arrs.h"
#include "datas.h"
#include "linalg.h"
#include "losses.h"
#include "math.h"
#include "rand.h"
#include "tensor.h"

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
    size_t num_imgs = s4.shape.front() * s4.shape[1];
    size_t output_img_size = s4.shape[2] * s4.shape.back();

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