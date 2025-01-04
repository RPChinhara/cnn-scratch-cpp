#include "arrs.h"
#include "rand.h"
#include "tensor.h"

int main () {
    tensor dl_dc3 = uniform_dist({3, 2, 4, 4}, 0.0f, 0.000001f);

    std::cout << dl_dc3 << "\n";
    std::cout << slice_4d(dl_dc3, 0, 1) << "\n";
    std::cout << slice_4d(dl_dc3, 1, 2) << "\n";
    std::cout << slice_4d(dl_dc3, 2, 1) << "\n";

    return 0;
}