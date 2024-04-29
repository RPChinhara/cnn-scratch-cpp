#pragma once

#include "dev.h"

class ten;

ten matmul(const ten &t_1, const ten &t_2, dev_type dev);
ten transpose(const ten &t);