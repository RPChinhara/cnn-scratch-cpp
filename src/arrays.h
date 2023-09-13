#pragma once

class Tensor;

Tensor clip_by_value(const Tensor& in, float clip_val_min, float clip_val_max);
Tensor slice(const Tensor& in, const unsigned int begin, const unsigned int size);