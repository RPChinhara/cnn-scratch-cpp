#include "random.h"
#include "tensor.h"

#include <random>
#include <cassert>

static std::mt19937 Rng()
{
    std::random_device rd;
    std::mt19937 rng(rd());
    return rng;
}

static void SetShape(Tensor& in, const std::vector<size_t>& shape)
{
    in.shape.reserve(shape.size());

    for (size_t elem : shape)
        assert(elem != 0);

    in.shape = std::move(shape);
}

static void SetSize(Tensor& in, const std::vector<size_t>& shape)
{
    if (in.shape.size() > 0) {
        size_t num_elem = 1;

        for (size_t elem : shape)
            num_elem *= elem;

        in.size = num_elem;
    } else {
        in.size = 1;
    }
}

static void SetNumChDim(Tensor& in,  const std::vector<size_t>& shape)
{
    if (in.shape.size() > 0) {
        in.num_ch_dim = 1;

        for (size_t i = 0; i < shape.size() - 1; ++i)
            in.num_ch_dim *= shape[i];

    } else {
        in.num_ch_dim = 0;
    }
}

Tensor NormalDistribution(const std::vector<size_t>& shape, const float mean, const float stddev)
{
    Tensor out = Tensor();
    SetShape(out, shape);
    SetSize(out, shape);
    out.elem = new float[out.size];

    std::normal_distribution<float> dist(mean, stddev);

    for (size_t i = 0; i < out.size; ++i)
        out[i] = dist(Rng());
    
    SetNumChDim(out, shape);
    return out;
}

Tensor Shuffle(const Tensor& in, const size_t random_state)
{
    Tensor out = in;
    std::mt19937 rng(random_state);

    for (size_t i = in.shape.front() - 1; i > 0; --i) {
        std::uniform_int_distribution<> dist(0, i);
        int j = dist(rng);

        for (size_t k = 0; k < in.shape.back(); ++k) {
            float temp = out[(in.shape.back() - 1) * i + i + k];
            out[(in.shape.back() - 1) * i + i + k] = out[(in.shape.back() - 1) * j + j + k];
            out[(in.shape.back() - 1) * j + j + k] = temp;
        }
    }
    return out;
}

Tensor UniformDistribution(const std::vector<size_t>& shape, const float min_val, const float max_val)
{
    Tensor out = Tensor();
    SetShape(out, shape);
    SetSize(out, shape);
    out.elem = new float[out.size];

    std::uniform_real_distribution<> dist(min_val, max_val);

    for (size_t i = 0; i < out.size; ++i)
        out[i] = dist(Rng());
    
    SetNumChDim(out, shape);
    return out;
}