#pragma once

#include "dev.h"

class tensor;

tensor matmul(const tensor &t1, const tensor &t2, dev_type dev);
tensor transpose(const tensor &t);