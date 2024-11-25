#pragma once

class tensor;

tensor argmax(const tensor& t);
tensor exp(const tensor& t);
tensor max(const tensor& t, const size_t axis);
tensor min(const tensor& t);
tensor sqrt(const tensor& x);
tensor square(const tensor& t);
tensor sum(const tensor& t, const size_t axis);