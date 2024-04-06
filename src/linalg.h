#pragma once

#include "dev.h"

class ten;

ten matmul(const ten &tensor1, const ten &tensor2, dev_type dev);
ten transpose(const ten &t);