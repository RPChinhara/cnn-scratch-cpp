#pragma once

class tensor;

tensor hyperbolic_tangent(const tensor &z_t);
tensor sigmoid(const tensor &t);

tensor relu_derivative(const tensor &z);
tensor sigmoid_derivative(const tensor &z);