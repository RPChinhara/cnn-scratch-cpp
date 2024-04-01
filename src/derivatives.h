#pragma once

#include "activations.h"

class Tensor;

Tensor dl_da_da_dz(const Tensor &y_target, const Tensor &y_pred, Act act);
Tensor dmse_dsigmoid_dsigmoid_dz(const Tensor &y_target, const Tensor &y_pred);
Tensor da_dz(const Tensor &tensor, Act act);