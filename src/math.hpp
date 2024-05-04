#pragma once

#include "dev.h"

class ten;

ten argmax(const ten &t);
ten Exp(const ten &t, dev_type dev);
ten Log(const ten &t, dev_type dev);
ten Max(const ten &t, const size_t axis);
ten Min(const ten &t);
ten sum(const ten &t, const size_t axis);