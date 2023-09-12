#pragma once

#include "types.h"

class Tensor;

Tensor shuffle(const Tensor& in, const u32 random_state);