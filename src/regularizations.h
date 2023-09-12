#pragma once

#include "types.h"

class Tensor;

void dropout(const f32 rate, const Tensor& in);
f32 l1(const f32 lambda, const Tensor& weight);
f32 l2(const f32 lambda, const Tensor& weight);