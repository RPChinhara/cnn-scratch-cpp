#include "arrs.h"
#include "linalg.h"
#include "tensor.h"

int main () {
    tensor x1 = variable({2, 2, 3}, {1,  2,  3,
                                     4,  5,  6,

                                     7,  8,  9,
                                     10, 11, 12});

    tensor x2 = variable({2, 3, 2}, {1,  2,
                                     3,  4,
                                     5,  6,

                                     7,  8,
                                     9,  10,
                                     11, 12});

    tensor x3 = variable({2, 3}, {1, 2, 3,
                                  4, 5, 6});

    tensor x4 = variable({3, 2}, {1, 2,
                                  3, 4,
                                  5, 6});

    std::cout << matmul(x1, x2) << "\n";
    std::cout << matmul(x3, x4) << "\n";

    return 0;
}