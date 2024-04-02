#pragma once

#include "dev.h"

class Ten;

Ten Argmax(const Ten &tensor);
Ten Exp(const Ten &tensor, Dev device);
Ten Log(const Ten &tensor, Dev device);
Ten Max(const Ten &tensor, const size_t axis);
Ten Min(const Ten &tensor);
Ten Sum(const Ten &tensor, const size_t axis);