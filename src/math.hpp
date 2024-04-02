#pragma once

#include "dev.h"

class Ten;

Ten Argmax(const Ten &ten);
Ten Exp(const Ten &ten, Dev dev);
Ten Log(const Ten &ten, Dev dev);
Ten Max(const Ten &ten, const size_t axis);
Ten Min(const Ten &ten);
Ten Sum(const Ten &ten, const size_t axis);