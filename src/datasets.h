#pragma once

#include "tensor.h"

struct AirPassengers {
    Tensor features;
    Tensor target;
};

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

AirPassengers load_air_passengers();
Cifar10       load_cifar10();
Imdb          load_imdb();
Iris          load_iris();
Mnist         load_mnist();