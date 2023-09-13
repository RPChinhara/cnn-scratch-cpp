#pragma once

class Tensor;

void dropout(const float rate, const Tensor& in);
float l1(const float lambda, const Tensor& weight);
float l2(const float lambda, const Tensor& weight);