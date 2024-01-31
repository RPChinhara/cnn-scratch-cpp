#pragma once

#include "device.h"

class Tensor;

Tensor MatMul(const Tensor &in1, const Tensor &in2, Device device);