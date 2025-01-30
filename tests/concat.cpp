#include "arrs.h"
#include "tensor.h"

int main () {
    tensor a = variable({2, 2}, {1, 2, 3, 4});
    tensor b = variable({2, 2}, {5, 6, 7, 8});
    tensor c = variable({2, 2}, {9, 10, 11, 12});
    tensor d = variable({2, 2}, {13, 14, 15, 16});

    std::cout << concat({a, b, c, d}, 1) << "\n";

    return 0;
}