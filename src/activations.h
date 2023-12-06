#pragma once

class Tensor;

Tensor Relu(const Tensor& in);
Tensor Sigmoid(const Tensor& in);
Tensor Softmax(const Tensor& in);
Tensor Softplus(const Tensor& in);