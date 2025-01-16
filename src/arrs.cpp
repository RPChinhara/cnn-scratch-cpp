#include "arrs.h"
#include "tensor.h"

#include <numeric>

tensor variable(const std::vector<size_t>& shape, const std::vector<float>& vals) {
    tensor t_new;
    t_new.shape = shape;
    t_new.size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    t_new.elems = new float[t_new.size];

    for (size_t i = 0; i < t_new.size; ++i)
        t_new[i] = vals[i];

    return t_new;
}

tensor fill(const std::vector<size_t>& shape, float val) {
    tensor t_new;
    t_new.shape = shape;
    t_new.size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    t_new.elems = new float[t_new.size];

    for (size_t i = 0; i < t_new.size; ++i)
        t_new[i] = val;

    return t_new;
}

tensor zeros(const std::vector<size_t>& shape) {
    tensor t_new;
    t_new.shape = shape;
    t_new.size = std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<size_t>());
    t_new.elems = new float[t_new.size]();

    return t_new;
}

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

tensor slice(const tensor& t, const size_t begin, const size_t size) {
    tensor t_new = zeros({size, t.shape.back()});

    for (auto i = begin * t.shape.back(); i < (begin * t.shape.back()) + (size * t.shape.back()); ++i)
        t_new[i - (begin * t.shape.back())] = t[i];

    return t_new;
}

tensor slice_3d(const tensor& t, const size_t begin, const size_t size) {
    tensor t_new = zeros({size, t.shape[1], t.shape.back()});

    for (size_t i = 0; i < size * t.shape[1] * t.shape.back(); ++i)
        t_new[i] = t[begin * t.shape[1] * t.shape.back() + i];

    return t_new;
}

tensor slice_4d(const tensor& t, const size_t begin, const size_t size) {
    tensor t_new = zeros({size, t.shape[1], t.shape[2], t.shape.back()});

    for (size_t i = 0; i < size * t.shape[1] * t.shape[2] * t.shape.back(); ++i)
        t_new[i] = t[begin * t.shape[1] * t.shape[2] * t.shape.back() + i];

    return t_new;
}

tensor slice_test(const tensor& t, const std::vector<size_t>& begin, const std::vector<size_t>& size) {
    int num_dims = t.shape.size();
    std::vector<int> strides(num_dims, 1);
    for (int i = num_dims - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * t.shape[i + 1];
    }

    // Compute flattened indices for the slice
    tensor t_new = zeros(size);
    std::vector<float> result;
    std::vector<size_t> current_indices = begin;
    int total_elements = 1;
    for (int s : size) total_elements *= s;

    size_t idx = 0;
    for (int i = 0; i < total_elements; ++i) {
        // Calculate the flat index for the current slice position
        int flat_index = 0;
        for (int d = 0; d < num_dims; ++d) {
            flat_index += current_indices[d] * strides[d];
        }
        // result.push_back(tensor_data[flat_index]);
        t_new[idx] = t[flat_index];
        ++idx;

        // Increment the current indices
        for (int d = num_dims - 1; d >= 0; --d) {
            current_indices[d]++;
            if (current_indices[d] < begin[d] + size[d]) break;
            current_indices[d] = begin[d]; // Reset dimension
        }
    }

    return t_new;
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

tensor broadcast_to(const tensor& t, const std::vector<size_t>& shape) {
    tensor t_new = zeros(shape);

    size_t idx = 0;

    if (t.shape.front() > t.shape.back()) {
        for (size_t i = 0; i < t_new.size; ++i) {
            if (i % t_new.shape.back() == 0 && i != 0)
                ++idx;

            t_new[i] = t[idx];
        }
    } else {
        for (size_t i = 0; i < t_new.size; ++i) {
            if (idx == t.shape.back())
                idx = 0;

            t_new[i] = t[idx];
            ++idx;
        }
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

// NOTE: Only supports 4D tensors!
tensor pad(const tensor& t, size_t pad_top, size_t pad_bottom, size_t pad_left, size_t pad_right) {
    size_t batch_size = t.shape.front();
    size_t depth = t.shape[1];
    size_t rows = t.shape[2];
    size_t cols = t.shape.back();

    size_t new_rows = rows + pad_top + pad_bottom;
    size_t new_cols = cols + pad_left + pad_right;

    tensor t_new = zeros({batch_size, depth, new_rows, new_cols});

    size_t num_mats = batch_size * depth;

    for (size_t b = 0; b < num_mats; ++b) {
        auto mat = slice(t, b * rows, rows);
        tensor new_mat = zeros({new_rows, new_cols});

        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                new_mat(i + pad_top, j + pad_left) = mat(i, j);
            }
        }

        for (size_t i = 0; i < new_mat.size; ++i)
                t_new[b * new_mat.size + i] = new_mat[i];
    }

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

std::pair<tensor, tensor> split(const tensor& x, const float test_size) {
    tensor x_train = zeros({static_cast<size_t>(std::floorf(x.shape.front() * (1.0 - test_size))), x.shape.back()});
    tensor x_test = zeros({static_cast<size_t>(std::ceilf(x.shape.front() * test_size)), x.shape.back()});

    for (auto i = 0; i < x_train.size; ++i)
        x_train[i] = x[i];

    for (auto i = x_train.size; i < x.size; ++i)
        x_test[i - x_train.size] = x[i];

    return std::make_pair(x_train, x_test);
}