#pragma once

class tensor;

tensor da_dz(const tensor &a);
tensor dl_da_da_dz(const tensor &y_true, const tensor &y_pred);