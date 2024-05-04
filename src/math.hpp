#pragma once

#include "dev.h"

class ten;

ten argmax(const ten &t);
ten exp(const ten &t, dev_type dev);
ten log(const ten &t, dev_type dev);
ten max(const ten &t, const size_t axis);
ten Min(const ten &t);
ten sum(const ten &t, const size_t axis);