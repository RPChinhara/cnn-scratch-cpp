#pragma once

class Tensor;

float CategoricalAccuracy(const Tensor &yTrue, const Tensor &yPred);