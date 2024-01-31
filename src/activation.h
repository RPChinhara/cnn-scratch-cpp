#pragma once

#include "device.h"

class Tensor;

Tensor Relu(const Tensor &in, Device device);
Tensor Softmax(const Tensor &in);