#include "arrs.h"

int main() {

    auto a = uniform_dist({3, 2, 2}, 0.0f, 0.0000001f);

    std::cout << slice_3d(a, 0, 1) << "\n";
    std::cout << slice_3d(a, 0, 2) << "\n";
    std::cout << slice_3d(a, 0, 3) << "\n";
    std::cout << slice_3d(a, 1, 2) << "\n";
    std::cout << slice_3d(a, 2, 3) << "\n";

    std::cout << slice(a, 0, 1) << "\n";
    std::cout << slice(a, 0, 2) << "\n";
    std::cout << slice(a, 0, 3) << "\n";

    std::cout << a << "\n";

    return 0;
}