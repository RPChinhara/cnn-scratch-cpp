#pragma once

#include "tensor.h"

#include <string>
#include <vector>

struct imdb {
    tensor x;
    tensor y;
};

struct iris {
    tensor x;
    tensor y;
};

struct mnist {
    tensor trainImages;
    tensor trainLabels;
    tensor testImages;
    tensor testLabels;
};

tensor load_aapl();
imdb load_imdb();
iris load_iris();
mnist load_mnist();