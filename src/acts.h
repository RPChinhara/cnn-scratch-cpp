#pragma once

class tensor;

tensor hyperbolic_tangent(const tensor& x);
tensor relu(const tensor& x);
tensor sigmoid(const tensor& x);
tensor softmax(const tensor& x);

tensor relu_derivative(const tensor& x);
tensor sigmoid_derivative(const tensor& x);