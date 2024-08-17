#pragma once

#include "dev.h"

class tensor;

tensor argmax(const tensor &t);
tensor exp(const tensor &t, dev_type dev);
tensor log(const tensor &t, dev_type dev);
tensor max(const tensor &t, const size_t axis);
tensor min(const tensor &t);
tensor sqrt(const tensor &x);
tensor sum(const tensor &t, const size_t axis);