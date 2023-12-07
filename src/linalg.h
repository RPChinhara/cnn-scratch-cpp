#pragma once

class Tensor;

Tensor MatMul(const Tensor& in1, const Tensor& in2);
Tensor Transpose(const Tensor& in);