#include "lyrs.h"
#include "arrs.h"
#include "dev.h"
#include "diffs.h"
#include "linalg.h"
#include "math.hpp"
#include "preproc.h"
#include "rd.h"
#include "ten.h"

#include <cassert>
#include <chrono>

cnn2d::cnn2d(const std::vector<size_t> &filters, float const lr) {
    this->filters = filters;
    this->lr = lr;
}

void cnn2d::train(const ten &xTrain, const ten &yTrain, const ten &xVal, const ten &yVal) {
    // ten kernel = zeros({3, 3});
    ten kernel = ten({3, 3}, {1, -1, 1, 0, 1, 0, -1, 0, 1});

    size_t kernelHeight = kernel.shape.front();
    size_t kernelWidth = kernel.shape.back();

    size_t inputHeight = xTrain.shape[1];
    size_t inputWidth = xTrain.shape[2];

    size_t outputHeight = inputHeight - kernelHeight + 1;
    size_t outputWidth = inputWidth - kernelWidth + 1;

    ten output = zeros({outputHeight, outputWidth});

    // size_t idx = 0;

    // for (auto i = 0; i < outputHeight; ++i)
    // {
    //     for (auto j = 0; i < outputWidth; ++j)
    //     {
    //         // ouput[idx] =
    //     }
    // }

    // std::cout << output << std::endl;
}

void cnn2d::predict(const ten &xTest, const ten &yTest) {
}

std::vector<ten> cnn2d::forward(const ten &input, const std::vector<ten> &kernel, const size_t stride) {
    std::vector<ten> weights;

    return weights;
}

gru::gru(const size_t units) {
}

std::pair<std::vector<ten>, std::vector<ten>> gru::init_params() {
    w_z = normal_dist({num_ins, num_hiddens});
    w_r = normal_dist({num_ins, num_hiddens});
    w_h = normal_dist({num_ins, num_hiddens});

    u_z = normal_dist({num_hiddens, num_hiddens});
    u_r = normal_dist({num_hiddens, num_hiddens});
    u_h = normal_dist({num_hiddens, num_hiddens});

    b_z = zeros({1, num_hiddens});
    b_r = zeros({1, num_hiddens});
    b_h = zeros({1, num_hiddens});

    h = zeros({batch_size, num_hiddens});
}

std::vector<ten> gru::forward(const ten &x) {
    init_params();

    auto z = act(matmul(x, w_z, GPU) + matmul(u_z, h, GPU) + b_z, SIGMOID, CPU);
    auto r = act(matmul(x, w_r, GPU) + matmul(u_r, h, GPU) + b_z, SIGMOID, CPU);
    auto h_tilde = act(matmul(x, w_h, GPU) + matmul(u_h, r * h, GPU) + b_z, TANH, CPU);
    h = (1 - z) * h + z * h_tilde;
}

nn::nn(const std::vector<size_t> &lyrs, const std::vector<act_type> &act_types, const float lr, std::function<float(const ten&, const ten&)> loss, std::function<float(const ten&, const ten&)> metric) {
    this->lyrs = lyrs;
    this->act_types = act_types;
    this->lr = lr;
    this->loss = loss;
    this->metric = metric;

    w_b = init_params();
    w_b_mom = init_params();
}

void nn::train(const ten &x_train, const ten &y_train, const ten &x_val, const ten &y_val) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        if (10 <= i && i < 20)
            lr = 0.009f;
        else if (20 <= i && i < 30)
            lr = 0.005f;
        else if (30 <= i)
            lr = 0.001f;

        std::random_device rd;
        auto rd_state = rd();

        ten x_shuffled = shuffle(x_train, rd_state);
        ten y_shuffled = shuffle(y_train, rd_state);

        ten x_batch;
        ten y_batch;

        for (auto j = 0; j < x_train.shape.front(); j += batch_size) {
            assert(0 < batch_size && batch_size <= x_train.shape.front());

            if (x_train.shape.front() <= j + batch_size) {
                x_batch = slice(x_shuffled, j, x_train.shape.front() - j);
                y_batch = slice(y_shuffled, j, x_train.shape.front() - j);
            } else {
                x_batch = slice(x_shuffled, j, batch_size);
                y_batch = slice(y_shuffled, j, batch_size);
            }

            a = forward(x_batch, w_b.first, w_b.second);

            std::vector<ten> dl_dz, dl_dw, dl_db;

            for (auto k = lyrs.size() - 1; 0 < k; --k) {
                if (k == lyrs.size() - 1)
                    dl_dz.push_back(dl_da_da_dz(y_batch, a.back(), act_types.back()));
                else
                    dl_dz.push_back(matmul(dl_dz[(lyrs.size() - 2) - k], transpose(w_b.first[k]), CPU) * da_dz(a[k - 1], act_types[k - 1]));

                if (k == 1)
                    dl_dw.push_back(matmul(transpose(x_batch), dl_dz[(lyrs.size() - 1) - k], CPU));
                else
                    dl_dw.push_back(matmul(transpose(a[k - 2]), dl_dz[(lyrs.size() - 1) - k], CPU));

                dl_db.push_back(sum(dl_dz[(lyrs.size() - 1) - k], 0));

                dl_dw[(lyrs.size() - 1) - k] = clip_by_value(dl_dw[(lyrs.size() - 1) - k], -grad_clip_threshold, grad_clip_threshold);
                dl_db[(lyrs.size() - 1) - k] = clip_by_value(dl_db[(lyrs.size() - 1) - k], -grad_clip_threshold, grad_clip_threshold);

                w_b_mom.first[k - 1] = mom * w_b_mom.first[k - 1] - lr * dl_dw[(lyrs.size() - 1) - k];
                w_b_mom.second[k - 1] = mom * w_b_mom.second[k - 1] - lr * dl_db[(lyrs.size() - 1) - k];

                w_b.first[k - 1] += w_b_mom.first[k - 1];
                w_b.second[k - 1] += w_b_mom.second[k - 1];
            }

            dl_dz.clear(), dl_dw.clear(), dl_db.clear();
        }

        std::vector<ten> a_val = forward(x_val, w_b.first, w_b.second);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "Epoch " << i << "/" << epochs << std::endl;
        std::cout << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << loss(y_batch, a.back()) << " - accuracy: " << metric(y_batch, a.back());
        std::cout << " - val_loss: " << loss(y_val, a_val.back()) << " - val_accuracy: " << metric(y_val, a_val.back()) << std::endl;
    }
}

float nn::evaluate(const ten &x, const ten &y) {
    a = forward(x, w_b.first, w_b.second);

    return loss(y, a.back());
}

ten nn::predict(const ten &x) {
    a = forward(x, w_b.first, w_b.second);

    return a.back();
}

std::pair<std::vector<ten>, std::vector<ten>> nn::init_params() {
    std::vector<ten> w;
    std::vector<ten> b;

    for (auto i = 0; i < lyrs.size() - 1; ++i) {
        w.push_back(normal_dist({lyrs[i], lyrs[i + 1]}, 0.0f, 0.2f));
        b.push_back(zeros({1, lyrs[i + 1]}));
    }

    return std::make_pair(w, b);
}

std::vector<ten> nn::forward(const ten &x, const std::vector<ten> &w, const std::vector<ten> &b) {
    std::vector<ten> a;

    for (auto i = 0; i < lyrs.size() - 1; ++i) {
        if (i == 0) {
            // (64, 10) -> (64, 1) or (64, 10) I think latter is clearer, but
            // former is more performant. (10, 64) -> (1, 64) x.T = (4, 10), w1
            // = (64, 4), w2 = (10(must), 64), w3 = (64, 3), output = (64, 3) x
            // = (10, 4), w1 = (4, 64), w2 = (64, 64), w3 = (64, 3), ouput =
            // (10, 3)

            ten z = matmul(x, w[i], CPU) + b[i];
            a.push_back(act(z, act_types[i], CPU));
        } else {
            ten z = matmul(a[i - 1], w[i], CPU) + b[i];
            a.push_back(act(z, act_types[i], CPU));
        }
    }

    return a;
}

rnn::rnn(const size_t lr, std::function<float(const ten&, const ten&)> loss) {
    this->lr = lr;
    this->loss = loss;

    w_hx = uniform_dist({hidden_size, in_size});
    w_hh = uniform_dist({hidden_size, hidden_size});
    w_hy = uniform_dist({out_size, hidden_size});

    b_h = zeros({hidden_size, batch_size});
    b_y = zeros({out_size, batch_size});
}

void rnn::train(const ten &x_train, const ten &y_train, const ten &x_val, const ten &y_val) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto a = forward(x_train);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << loss(y_train, transpose(a.back())) << std::endl;
    }
}

std::vector<ten> rnn::forward(const ten &x) {
    ten h_t = zeros({hidden_size, batch_size});
    // ten y_t;

    std::vector<ten> h;
    std::vector<ten> y;

    for (auto i = 0; i < seq_length; ++i) {
        size_t idx = i;
        ten x_t = zeros({batch_size, in_size});

        // for (auto i = 0; i < batch_size * num_features; ++i)
        for (auto i = 0; i < batch_size; ++i) {
            ten features;

            features = slice(x, idx, 1);
            idx += seq_length;

            x_t[i] = features[0];
        }

        // (now) 50 1, 1 8316 = 50 8316 -> 50 50, 50 8316 = 50 8316 -> 1 50, 50
        // 8316 = 1 8316

        // 8316 1, 1 50 = 8316 50 -> 8316 50, 50 50 = 8316 50 -> 8316 50, 50 1 =
        // 8316 1

        // 50 8316, 8316 1 = 50 1 -> 50 50, 50 1 = 50 1 -> 1 50, 50 1 = 1 1
        // I think this is wrong because when you think about it it's weird that
        // getting only one ouput even thougth I input 8316 batches.

        h_t = act(matmul(w_hx, transpose(x_t), CPU) + matmul(w_hh, h_t, CPU) + b_h, TANH, GPU);
        ten y_t = matmul(w_hy, h_t, CPU) + b_y;

        h.push_back(h_t);
        y.push_back(y_t);
    }

    return y;
}

ten embedding(const size_t vocab_size, const size_t cols, const ten &ind) {
    for (auto i = 0; i < ind.size; ++i)
        assert(ind[i] < vocab_size);

    ten embeddings_mat = uniform_dist({vocab_size, cols});

    std::cout << embeddings_mat << std::endl;

    ten dense_vecs = zeros({ind.shape.front(), ind.shape.back(), cols});

    for (auto i = 0; i < ind.size; ++i) {
        auto a = slice(embeddings_mat, ind[i], 1);

        for (auto j = 0; j < a.size; ++j)
            dense_vecs[cols * i + j] = a[j];
    }

    return dense_vecs;
}