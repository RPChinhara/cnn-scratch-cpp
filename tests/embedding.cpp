#include "arrs.h"
#include "tensor.h"
#include "lyrs.h"

#include <iostream>

int main() {
    auto a = variable({2, 3}, {1, 2, 3,
                               1, 2, 3});
    auto v = embedding(10, 3, a);

    std::cout << v.mat << "\n";
    std::cout << v.dense_vecs << "\n";
}