#include "arrs.h"
#include "tensor.h"
#include "lyrs.h"

#include <iostream>

int main() {
    size_t vocab_size = 10;
    size_t embedding_dim = 3;

    auto input = variable({2, 3}, {0, 2, 3,

                                   7, 4, 9});

    auto lyr = embedding(vocab_size, embedding_dim);
    tensor embedded_tokens = lyr.adapt(input);

    std::cout << lyr.embedding_mat << "\n";
    std::cout << embedded_tokens << "\n";
}