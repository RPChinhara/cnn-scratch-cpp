#pragma once

#include "dev.h"

class ten;

ten matmul(const ten &t1, const ten &t2, dev_type dev);
ten transpose(const ten &t);