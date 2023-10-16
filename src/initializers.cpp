#include "initializers.h"
#include "tensor.h"

#include <cassert>
#include <random>

static std::mt19937 gen() {
    std::random_device rd;
    std::mt19937 gen(rd());
    return gen;
}

static void set_shape(Tensor& in, const std::vector<unsigned int>& shape) {
    in._shape.reserve(shape.size());
    for (unsigned int elem : shape)
        assert(elem != 0);

    in._shape = std::move(shape);
}

static void set_size(Tensor& in, const std::vector<unsigned int>& shape) {
    if (in._shape.size() > 0) {
        unsigned int num_elem = 1;
        for (unsigned int elem : shape)
            num_elem *= elem;

        in._size = num_elem;
    } else {
        in._size = 1;
    }
}

static void set_num_ch_dim(Tensor& in,  const std::vector<unsigned int>& shape) {
    if (in._shape.size() > 0) {
        in._num_ch_dim = 1;
        for (int i = 0; i < shape.size() - 1; ++i)
            in._num_ch_dim *= shape[i];

    } else {
        in._num_ch_dim = 0;
    }
}

Tensor glorot_normal_distribution(const std::vector<unsigned int>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new float[out._size];

    auto it    = shape.begin();
    auto next  = ++it;
    float stddev = sqrt(2.0f / (shape.front() + *next));
    std::normal_distribution<float> dist(0.0f, stddev);

    for (unsigned int i = 0; i < out._size; ++i)
        out[i] = dist(gen());

    set_num_ch_dim(out, shape);
    return out;
}

Tensor glorot_uniform_distribution(const std::vector<unsigned int>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new float[out._size];

    auto it   = shape.begin();
    auto next = ++it;
    float limit = sqrt(6.0f / (shape.front() + *next));
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (unsigned int i = 0; i < out._size; ++i)
        out[i] = dist(gen());

    set_num_ch_dim(out, shape);
    return out;
}

Tensor he_normal_distribution(const std::vector<unsigned int>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new float[out._size];

    float stddev = sqrt(2.0f / shape.front());
    std::normal_distribution<float> dist(0.0f, stddev);

    for (unsigned int i = 0; i < out._size; ++i)
        out[i] = dist(gen());

    set_num_ch_dim(out, shape);
    return out;
}

Tensor he_uniform_distribution(const std::vector<unsigned int>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new float[out._size];

    float limit = sqrt(6.0f / shape.front());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (unsigned int i = 0; i < out._size; ++i)
        out[i] = dist(gen());

    set_num_ch_dim(out, shape);
    return out;
}

Tensor lecun_normal_distribution(const std::vector<unsigned int>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new float[out._size];

    float stddev = sqrt(1.0f / shape.front());
    std::normal_distribution<float> dist(0.0f, stddev);

    for (unsigned int i = 0; i < out._size; ++i)
        out[i] = dist(gen());

    set_num_ch_dim(out, shape);
    return out;
}

Tensor lecun_uniform_distribution(const std::vector<unsigned int>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new float[out._size];

    float limit = sqrt(3.0f / shape.front());
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (unsigned int i = 0; i < out._size; ++i)
        out[i] = dist(gen());

    set_num_ch_dim(out, shape);
    return out;
}

Tensor normal_distribution(const std::vector<unsigned int>& shape, const float mean, const float stddev) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new float[out._size];

    std::normal_distribution<float> dist(mean, stddev);

    for (unsigned int i = 0; i < out._size; ++i)
        out[i] = dist(gen());
    
    set_num_ch_dim(out, shape);
    return out;
}

static void set_elem(Tensor& out, const float value) {
    out._elem = new float[out._size];
    std::fill(out._elem, out._elem + out._size, value);
}

Tensor ones(const std::vector<unsigned int>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    set_elem(out, 1.0f);
    set_num_ch_dim(out, shape);
    return out;
}

Tensor uniform_distribution(const std::vector<unsigned int>& shape, const float min_val, const float max_val) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new float[out._size];

    std::uniform_real_distribution<> dist(min_val, max_val);

    for (unsigned int i = 0; i < out._size; ++i)
        out[i] = dist(gen());
    
    set_num_ch_dim(out, shape);
    return out;
}

Tensor zeros(const std::vector<unsigned int>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    set_elem(out, 0.0f);
    set_num_ch_dim(out, shape);
    return out;
}