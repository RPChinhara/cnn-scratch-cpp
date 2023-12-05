#pragma once

class Tensor;

Tensor argmax(const Tensor& in);
Tensor exp(const Tensor& in);
Tensor log(const Tensor& in);
Tensor max(const Tensor& in, const unsigned short axis);
Tensor maximum(const Tensor& in1, const Tensor& in2);
Tensor min(const Tensor& in);
Tensor square(const Tensor& in);
Tensor sum(const Tensor& in, const unsigned short axis);
Tensor tanh(const Tensor& in);
Tensor variance(const Tensor& in);