#include "arrs.h"
#include "tensor.h"

#include <numeric>

tensor clip_by_value(const tensor& t, float clip_val_min, float clip_val_max) {
    tensor t_new = t;

    for (auto i = 0; i < t.size; ++i) {
        if (t[i] < clip_val_min)
            t_new[i] = clip_val_min;
        else if (clip_val_max < t[i])
            t_new[i] = clip_val_max;
    }

    return t_new;
}

tensor one_hot(const tensor& t, const size_t depth) {
    tensor t_new = zeros({t.size, depth});

    for (size_t i = 0; i < t.size; ++i) {
        size_t index = t[i] + (i * depth);
        t_new[index] = 1.0f;
    }

    return t_new;
}

tensor slice(const tensor& t, const size_t begin, const size_t size) {
    tensor t_new = zeros({size, t.shape.back()});

    for (auto i = begin * t.shape.back(); i < (begin * t.shape.back()) + (size * t.shape.back()); ++i)
        t_new[i - (begin * t.shape.back())] = t[i];

    return t_new;
}

std::pair<tensor, tensor> split(const tensor& x, const float test_size) {
    tensor x_train = zeros({static_cast<size_t>(std::floorf(x.shape.front() * (1.0 - test_size))), x.shape.back()});
    tensor x_test = zeros({static_cast<size_t>(std::ceilf(x.shape.front() * test_size)), x.shape.back()});

    for (auto i = 0; i < x_train.size; ++i)
        x_train[i] = x[i];

    for (auto i = x_train.size; i < x.size; ++i)
        x_test[i - x_train.size] = x[i];

    return std::make_pair(x_train, x_test);
}

tensor vslice(const tensor& t, const size_t col) {
    tensor t_new = zeros({t.shape.front(), t.shape.back() - 1});

    std::vector<float> new_elems;
    for (auto i = 0; i < t.size; ++i) {
        size_t current_col = i % t.shape.back();
        if (current_col == col)
            continue;
        else
            new_elems.push_back(t[i]);
    }

    for (auto i = 0; i < new_elems.size(); ++i)
        t_new[i] = new_elems[i];

    return t_new;
}

tensor vstack(const std::vector<tensor>& ts) {
    size_t first_dim = ts.front().shape.back();

    size_t num_rows = 0;

    for (auto i = 0; i < ts.size(); ++i)
        num_rows += ts[i].shape.front();

    tensor t_new = zeros({num_rows, ts.front().shape.back()});

    size_t idx = 0;
    for (auto i = 0; i < ts.size(); ++i) {
        for (auto j = 0; j < ts[i].size; ++j) {
            t_new[idx] = ts[i][j];
            ++idx;
        }
    }

    return t_new;
}

tensor zeros(const std::vector<size_t>& shape) {
    tensor t_new = tensor();

    t_new.shape = shape;

    if (0 < t_new.shape.size())
        t_new.size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    else
        t_new.size = 1;

    t_new.elems = new float[t_new.size];
    std::fill(t_new.elems, t_new.elems + t_new.size, 0.0f);

    return t_new;
}