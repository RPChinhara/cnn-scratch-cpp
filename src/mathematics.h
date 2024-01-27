#pragma once

class Tensor;

Tensor Argmax(const Tensor& in);
Tensor Exp(const Tensor& in);
Tensor Log(const Tensor& in);
Tensor Max(const Tensor& in, const size_t axis);
Tensor Min(const Tensor& in);
Tensor Sum(const Tensor& in, const size_t axis);