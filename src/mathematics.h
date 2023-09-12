#pragma once

#include "types.h"

class Tensor;

Tensor argmax(const Tensor& in);
Tensor exp(const Tensor& in);
Tensor log(const Tensor& in);
Tensor max(const Tensor& in, const u16 axis);
Tensor maximum(const Tensor& in1, const Tensor& in2);
Tensor mean(const Tensor& in);
Tensor min(const Tensor& in);
Tensor square(const Tensor& in);
Tensor sum(const Tensor& in, const u16 axis);
Tensor variance(const Tensor& in);