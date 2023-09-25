#pragma once

#include "tensor.h"

struct Cifar10 {
    uint8_t label;
    uint8_t data[3072]; // 32x32x3 bytes
};

struct Imdb {
    Tensor features;
    Tensor target;
};

struct Iris {
    Tensor features;
    Tensor target;
};

struct Mnist {
    Tensor features;
    Tensor target;
};

Tensor load_air_passengers();
Cifar10 load_cifar10();
Imdb load_imdb();
Iris load_iris(); // TODO: I could return like std::pair<Tensor, Tensor>? Which is better returning by class or this?
Mnist load_mnist();