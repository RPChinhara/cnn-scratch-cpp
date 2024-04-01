#pragma once

#include "acts.h"

class Tensor;

Tensor dl_da_da_dz(const Tensor &y_target, const Tensor &y_pred, Act act);
Tensor da_dz(const Tensor &tensor, Act act);