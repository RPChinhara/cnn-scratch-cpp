#include "arrs.h"

int main () {
    tensor dl_dc3 = uniform_dist({3, 2, 4, 4}, 0.0f, 0.000001f);
    tensor dl_dc3_test = uniform_dist({2, 4, 4}, 0.0f, 0.000001f);

    std::cout << slice_test(dl_dc3_test, {0, 0, 0}, {1, 4, 4}) << "\n";
    std::cout << slice_test(dl_dc3, {0, 0, 0, 0}, {1, 1, 4, 4}) << "\n";
    std::cout << slice_test(dl_dc3, {0, 0, 0, 0}, {1, 1, 10, 10}) << "\n";

    return 0;
}