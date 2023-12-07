#pragma once

class Tensor;

Tensor Argmax(const Tensor& in);
Tensor Exp(const Tensor& in);
Tensor Log(const Tensor& in);
Tensor Max(const Tensor& in, const unsigned short axis);
Tensor Maximum(const Tensor& in1, const Tensor& in2);
Tensor Min(const Tensor& in);
Tensor Square(const Tensor& in);
Tensor Sum(const Tensor& in, const unsigned short axis);
Tensor Tanh(const Tensor& in);