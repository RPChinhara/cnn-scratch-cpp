#include "random.h"
#include "tensor.h"

#include <random>

Tensor shuffle(const Tensor& in, const unsigned int random_state) {
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