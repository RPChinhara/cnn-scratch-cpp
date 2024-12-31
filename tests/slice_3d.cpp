#include "arrs.h"
#include "rand.h"
#include "tensor.h"

int main() {

    auto a = uniform_dist({12, 2, 2}, 0.0f, 0.0000001f);

    std::cout << slice_3d(a, 0, 1) << "\n";
    std::cout << slice_3d(a, 0, 2) << "\n";
    std::cout << slice_3d(a, 0, 3) << "\n";

    std::cout << slice_3d(a, 1, 2) << "\n";
    std::cout << slice_3d(a, 2, 3) << "\n";

    std::cout << slice_3d(a, 0, 4) << "\n";
    std::cout << slice_3d(a, 4, 4) << "\n";
    std::cout << slice_3d(a, 8, 4) << "\n";

    std::cout << a << std::endl;

    return 0;
}