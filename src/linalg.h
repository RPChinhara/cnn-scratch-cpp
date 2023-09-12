#pragma once

class Tensor;

Tensor matmul(const Tensor& in1, const Tensor& in2);
Tensor transpose(const Tensor& in);