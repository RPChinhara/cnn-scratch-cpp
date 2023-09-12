#pragma once

#include "types.h"

class Tensor;

Tensor clip_by_value(const Tensor& in, f32 clip_val_min, f32 clip_val_max);
Tensor slice(const Tensor& in, const u32 begin, const u32 size);