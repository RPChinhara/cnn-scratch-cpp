#pragma once

class Tensor;

Tensor relu(const Tensor& in);
Tensor sigmoid(const Tensor& in);
Tensor softmax(const Tensor& in);
Tensor softplus(const Tensor& in);