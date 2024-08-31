#pragma once

#include "dev.h"

class tensor;

tensor argmax(const tensor &t);
tensor exp(const tensor &t);
tensor log(const tensor &t);
tensor max(const tensor &t, const size_t axis);
tensor min(const tensor &t);
tensor sqrt(const tensor &x);
tensor sum(const tensor &t, const size_t axis);