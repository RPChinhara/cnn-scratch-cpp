#include "arrs.h"
#include "lyrs.h"

int main () {
    tensor foo = variable({2, 4}, {10, 11, 12, 13, 0.2, 0.3, 0.2, 0.111});

    tensor output = layer_normalization(foo);

    std::cout << output << "\n";

    return 0;
}