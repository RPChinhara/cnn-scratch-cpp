#pragma once

class Tensor;

Tensor MatMul(const Tensor& in_1, const Tensor& in_2);
Tensor Transpose(const Tensor& in);