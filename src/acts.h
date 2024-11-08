#pragma once

class tensor;

tensor hyperbolic_tangent(const tensor &z_t);
tensor sigmoid(const tensor &t);
tensor softmax(const tensor &z);

tensor relu_derivative(const tensor &z);
tensor sigmoid_derivative(const tensor &z);