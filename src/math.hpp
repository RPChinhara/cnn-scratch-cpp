#pragma once

#include "dev.h"

class Ten;

Ten Argmax(const Ten &t);
Ten Exp(const Ten &t, Dev dev);
Ten Log(const Ten &t, Dev dev);
Ten Max(const Ten &t, const size_t axis);
Ten Min(const Ten &t);
Ten Sum(const Ten &t, const size_t axis);