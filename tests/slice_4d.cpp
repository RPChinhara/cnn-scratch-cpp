#include "arrs.h"

int main () {
    tensor dl_dc3 = uniform_dist({3, 2, 4, 4}, 0.0f, 0.000001f);

    std::cout << dl_dc3 << "\n";
    std::cout << slice_4d(dl_dc3, 0) << "\n";
    std::cout << slice_4d(dl_dc3, 1 * dl_dc3.shape[1] * dl_dc3.shape[2] * dl_dc3.shape.back()) << "\n";
    std::cout << slice_4d(dl_dc3, 2 * dl_dc3.shape[1] * dl_dc3.shape[2] * dl_dc3.shape.back()) << "\n";

    return 0;
}