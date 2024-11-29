#pragma once

class tensor;

tensor add(const tensor& x, const tensor& y);
tensor subtract(const tensor& x, const tensor& y);
tensor multiply(const tensor& x, const tensor& y);
tensor divide(const tensor& x, const tensor& y);

tensor exp(const tensor& x);
tensor sqrt(const tensor& x);
tensor square(const tensor& x);

tensor max(const tensor& t, const size_t axis);
tensor min(const tensor& t);
tensor sum(const tensor& t, const size_t axis);

tensor argmax(const tensor& t);