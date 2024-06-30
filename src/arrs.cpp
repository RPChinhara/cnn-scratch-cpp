#include "arrs.h"
#include "ten.h"

#include <cassert>
#include <numeric>

ten clip_by_value(const ten &t, float clip_val_min, float clip_val_max)
{
    assert(clip_val_min <= clip_val_max);

    ten t_new = t;

    for (auto i = 0; i < t.size; ++i)
    {
        if (t[i] < clip_val_min)
            t_new[i] = clip_val_min;
        else if (clip_val_max < t[i])
            t_new[i] = clip_val_max;
    }

    return t_new;
}

ten slice(const ten &t, const size_t begin, const size_t size)
{
    ten t_new = zeros({size, t.shape.back()});

    for (auto i = begin * t.shape.back(); i < (begin * t.shape.back()) + (size * t.shape.back()); ++i)
        t_new[i - (begin * t.shape.back())] = t[i];

    return t_new;
}

std::pair<ten, ten> split(const ten &x, const float test_size)
{
    ten x_train = zeros({static_cast<size_t>(std::floorf(x.shape.front() * (1.0 - test_size))), x.shape.back()});
    ten x_test = zeros({static_cast<size_t>(std::ceilf(x.shape.front() * test_size)), x.shape.back()});

    for (auto i = 0; i < x_train.size; ++i)
        x_train[i] = x[i];

    for (auto i = x_train.size; i < x.size; ++i)
        x_test[i - x_train.size] = x[i];

    return std::make_pair(x_train, x_test);
}

ten zeros(const std::vector<size_t> &shape)
{
    ten t_new = ten();

    for (auto i : shape)
        assert(i != 0);

    t_new.shape = shape;

    if (0 < t_new.shape.size())
        t_new.size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    else
        t_new.size = 1;

    t_new.elem = new float[t_new.size];
    std::fill(t_new.elem, t_new.elem + t_new.size, 0.0f);

    return t_new;
}