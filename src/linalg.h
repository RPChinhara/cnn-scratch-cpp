#pragma once

class Tensor;

enum Device
{
    CPU,
    GPU
};

Tensor MatMul(const Tensor& in1, const Tensor& in2, Device device);