#pragma once

#include "tensor.h"

#include <array>
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
    tensor train_imgs;
    tensor train_labels;
    tensor test_imgs;
    tensor test_labels;
};

tensor load_aapl();
std::array<std::vector<std::string>, 3> load_daily_dialog();
iris load_iris();
mnist load_mnist();