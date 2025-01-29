#include "arrs.h"
#include "tensor.h"

int main () {
    tensor x = variable({4, 4}, {1.0f,   2.0f,  3.0f,  4.0f,
                                 5.0f,   6.0f,  7.0f,  8.0f,
                                 9.0f,  10.0f, 11.0f, 12.0f,
                                 13.0f, 14.0f, 15.0f, 16.0f});

    std::cout << x << "\n";
    std::cout << x.slice(1, 3, 1, 3) << "\n";
    std::cout << x.slice_rows(1, 3) << "\n";
    std::cout << x.slice_cols(0, 2) << "\n";
    std::cout << x.slice_cols(2, 4) << "\n";

    return 0;
}