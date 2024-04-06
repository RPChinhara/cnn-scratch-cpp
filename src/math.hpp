#pragma once

#include "dev.h"

class ten;

ten Argmax(const ten &t);
ten Exp(const ten &t, dev_type dev);
ten Log(const ten &t, dev_type dev);
ten Max(const ten &t, const size_t axis);
ten Min(const ten &t);
ten Sum(const ten &t, const size_t axis);