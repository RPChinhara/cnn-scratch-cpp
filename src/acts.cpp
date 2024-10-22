#include "acts.h"
#include "math.hpp"
#include "tensor.h"

tensor sigmoid(const tensor &t) {
    tensor t_new = t;

    return 1.0 / (1.0 + exp(-t));
}