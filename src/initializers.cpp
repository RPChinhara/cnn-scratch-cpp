#include "initializers.h"
#include "tensor.h"

#include <cassert>
#include <random>

static std::mt19937 gen() {
    std::random_device rd;
    std::mt19937 gen(rd());
    return gen;
}

static void set_shape(Tensor& in, const std::vector<u32>& shape) {
    in._shape.reserve(shape.size());
    for (u32 elem : shape)
        assert(elem != 0);
    in._shape = std::move(shape);
}

static void set_size(Tensor& in, const std::vector<u32>& shape) {
    if (in._shape.size() > 0) {
        u32 num_elem = 1;
        for (u32 elem : shape)
            num_elem *= elem;
        in._size = num_elem;
    } else
        in._size = 1;
}

static void set_num_ch_dim(Tensor& in,  const std::vector<u32>& shape) {
    if (in._shape.size() > 0) {
        in._num_ch_dim = 1;
        for (s32 i = 0; i < shape.size() - 1; ++i)
            in._num_ch_dim *= shape[i];
    } else
        in._num_ch_dim = 0;
}

Tensor glorot_normal_distribution(const std::vector<u32>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new f32[out._size];

    auto it    = shape.begin();
    auto next  = ++it;
    f32 stddev = sqrt(2.0f / (shape.front() + *next));
    std::normal_distribution<f32> dist(0.0f, stddev);

    for (u32 i = 0; i < out._size; ++i)
        out[i] = dist(gen());
    set_num_ch_dim(out, shape);
    return out;
}

Tensor glorot_uniform_distribution(const std::vector<u32>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new f32[out._size];

    auto it   = shape.begin();
    auto next = ++it;
    f32 limit = sqrt(6.0f / (shape.front() + *next));
    std::uniform_real_distribution<f32> dist(-limit, limit);

    for (u32 i = 0; i < out._size; ++i)
        out[i] = dist(gen());
    set_num_ch_dim(out, shape);
    return out;
}

Tensor he_normal_distribution(const std::vector<u32>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new f32[out._size];

    f32 stddev = sqrt(2.0f / shape.front());
    std::normal_distribution<f32> dist(0.0f, stddev);

    for (u32 i = 0; i < out._size; ++i)
        out[i] = dist(gen());
    set_num_ch_dim(out, shape);
    return out;
}

Tensor he_uniform_distribution(const std::vector<u32>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new f32[out._size];

    f32 limit = sqrt(6.0f / shape.front());
    std::uniform_real_distribution<f32> dist(-limit, limit);

    for (u32 i = 0; i < out._size; ++i)
        out[i] = dist(gen());
    set_num_ch_dim(out, shape);
    return out;
}

Tensor lecun_normal_distribution(const std::vector<u32>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new f32[out._size];

    f32 stddev = sqrt(1.0f / shape.front());
    std::normal_distribution<f32> dist(0.0f, stddev);

    for (u32 i = 0; i < out._size; ++i)
        out[i] = dist(gen());
    set_num_ch_dim(out, shape);
    return out;
}

Tensor lecun_uniform_distribution(const std::vector<u32>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new f32[out._size];

    f32 limit = sqrt(3.0f / shape.front());
    std::uniform_real_distribution<f32> dist(-limit, limit);

    for (u32 i = 0; i < out._size; ++i)
        out[i] = dist(gen());
    set_num_ch_dim(out, shape);
    return out;
}

Tensor normal_distribution(const std::vector<u32>& shape, const f32 mean, const f32 stddev) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new f32[out._size];

    std::normal_distribution<f32> dist(mean, stddev);

    for (u32 i = 0; i < out._size; ++i)
        out[i] = dist(gen());
    set_num_ch_dim(out, shape);
    return out;
}

static void set_elem(Tensor& out, const f32 value) {
    out._elem = new f32[out._size];
    std::fill(out._elem, out._elem + out._size, value);
}

Tensor ones(const std::vector<u32>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    set_elem(out, 1.0f);
    set_num_ch_dim(out, shape);
    return out;
}

Tensor uniform_distribution(const std::vector<u32>& shape, const f32 min_val, const f32 max_val) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    out._elem = new f32[out._size];

    std::uniform_int_distribution<u32> dist(min_val, max_val);

    for (u32 i = 0; i < out._size; ++i)
        out[i] = dist(gen());
    set_num_ch_dim(out, shape);
    return out;
}

Tensor zeros(const std::vector<u32>& shape) {
    Tensor out = Tensor();
    set_shape(out, shape);
    set_size(out, shape);
    set_elem(out, 0.0f);
    set_num_ch_dim(out, shape);
    return out;
}