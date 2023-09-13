#include "random.h"
#include "tensor.h"

#include <random>

Tensor shuffle(const Tensor& in, const unsigned int random_state) {
    //   0   1   2   3 -> in_shape.back() - 1 x i + i + k = 0,  in_shape.back() - 1 x i + i + k = 1
    //   4   5   6   7 -> in_shape.back() - 1 x i + i + k = 4,  in_shape.back() - 1 x i + i + k = 5
    //   8   9  10  11 -> in_shape.back() - 1 x i + i + k = 8
    //  12  13  14  15 -> in_shape.back() - 1 x i + i + k = 12
    // ...
    //  84  85  86  87 -> in_shape.back() - 1 x j(21) + j(21) + k = 84, in_shape.back() - 1 x j(21) + j(21) + k = 85
    //  88  89  90  91
    //  92  93  94  95
    //  96  97  98  99

    Tensor out = in;
    std::mt19937 rng(random_state);

    for (unsigned int i = in._shape.front() - 1; i > 0; --i) {
        std::uniform_int_distribution<unsigned int> dist(0, i);
        unsigned int j = dist(rng);
        for (unsigned int k = 0; k < in._shape.back(); ++k) {
            float temp = out[(in._shape.back() - 1) * i + i + k];
            out[(in._shape.back() - 1) * i + i + k] = out[(in._shape.back() - 1) * j + j + k];
            out[(in._shape.back() - 1) * j + j + k] = temp;
        }
    }
    return out;
}