#include "arrs.h"
#include "tensor.h"
#include "lyrs.h"

#include <iostream>

int main() {
    size_t vocab_size = 10;
    size_t embedding_dim = 3;

    auto input = variable({2, 3}, {0, 2, 3,
                                   7, 4, 9});
                                   
    embedding lyr = embedding(vocab_size, embedding_dim, input);

    std::cout << lyr.mat << "\n";
    std::cout << lyr.dense_vecs << "\n";
}