#include "arrs.h"
#include "tensor.h"
#include "lyrs.h"

#include <iostream>

int main() {
    auto input = variable({2, 3}, {0, 2, 3,
                                   7, 4, 8});
    embedding lyr = embedding(10, 3, input);

    std::cout << lyr.mat << "\n";
    std::cout << lyr.dense_vecs << "\n";
}