#include "lyrs.h"
#include "arrs.h"
#include "dev.h"
#include "linalg.h"
#include "math.hpp"
#include "preproc.h"
#include "rd.h"
#include "tensor.h"

#include <cassert>
#include <chrono>

cnn2d::cnn2d(const std::vector<size_t> &filters, float const lr) {
    this->filters = filters;
    this->lr = lr;
}

void cnn2d::train(const tensor &xTrain, const tensor &yTrain, const tensor &xVal, const tensor &yVal) {
    // tensor kernel = zeros({3, 3});
    tensor kernel = tensor({3, 3}, {1, -1, 1, 0, 1, 0, -1, 0, 1});

    size_t kernelHeight = kernel.shape.front();
    size_t kernelWidth = kernel.shape.back();

    size_t inputHeight = xTrain.shape[1];
    size_t inputWidth = xTrain.shape[2];

    size_t outputHeight = inputHeight - kernelHeight + 1;
    size_t outputWidth = inputWidth - kernelWidth + 1;

    tensor output = zeros({outputHeight, outputWidth});

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

void cnn2d::predict(const tensor &xTest, const tensor &yTest) {
}

std::vector<tensor> cnn2d::forward(const tensor &input, const std::vector<tensor> &kernel, const size_t stride) {
    std::vector<tensor> weights;

    return weights;
}

gru::gru(const size_t units) {
}

std::pair<std::vector<tensor>, std::vector<tensor>> gru::init_params() {
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

std::vector<tensor> gru::forward(const tensor &x) {
    init_params();

    // auto z = act(matmul(x, w_z, GPU) + matmul(u_z, h, GPU) + b_z, SOFTMAX, CPU);
    // auto r = act(matmul(x, w_r, GPU) + matmul(u_r, h, GPU) + b_z, SOFTMAX, CPU);
    // auto h_tilde = act(matmul(x, w_h, GPU) + matmul(u_h, r * h, GPU) + b_z, SOFTMAX, CPU);
    // h = (1 - z) * h + z * h_tilde;
}

lstm::lstm(const size_t lr, loss_func loss) {
    this->lr = lr;
    this->loss = loss;

    w_xh = uniform_dist({hidden_size, in_size});
    w_hh = uniform_dist({hidden_size, hidden_size});
    w_hy = uniform_dist({out_size, hidden_size});

    b_h = zeros({hidden_size, batch_size});
    b_y = zeros({out_size, batch_size});
}

void lstm::train(const tensor &x_train, const tensor &y_train, const tensor &x_val, const tensor &y_val) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto a = forward(x_train);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << loss(y_train, a.back()) << std::endl;
    }
}

std::vector<tensor> lstm::forward(const tensor &x) {
    tensor h_t = zeros({hidden_size, batch_size});
    // tensor y_t;

    std::vector<tensor> h;
    std::vector<tensor> y;

    for (auto i = 0; i < seq_length; ++i) {
        size_t idx = i;
        tensor x_t = zeros({batch_size, in_size});

        // for (auto i = 0; i < batch_size * num_features; ++i)
        for (auto i = 0; i < batch_size; ++i) {
            tensor features;

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

        // h_t = activationmatmul(w_xh, transpose(x_t), CPU) + matmul(w_hh, h_t, CPU) + b_h, TANH, GPU);
        tensor y_t = matmul(w_hy, h_t, CPU) + b_y;

        h.push_back(h_t);
        y.push_back(y_t);
    }

    return y;
}

nn::nn(const std::vector<size_t> &lyrs, const std::vector<act_func> &activations, const loss_func &loss, const metric_func &metric, const float lr) {
    this->lyrs = lyrs;
    this->activations = activations;
    this->loss = loss;
    this->metric = metric;
    this->lr = lr;

    w_b = init_params();
    w_b_mom = init_params();
}

tensor da_dz(const tensor &a) {
    tensor t_new = a;

    for (auto i = 0; i < a.size; ++i)
    {
        if (0.0f < a[i])
            t_new[i] = 1.0f;
        else if (a[i] == 0.0f)
            t_new[i] = 0.0f;
        else
            t_new[i] = 0.0f;
    }

    return t_new;
}

tensor dl_da_da_dz(const tensor &y_true, const tensor &y_pred) {
    return (y_pred - y_true);
}

void nn::train(const tensor &x_train, const tensor &y_train, const tensor &x_val, const tensor &y_val) {
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

        tensor x_shuffled = shuffle(x_train, rd_state);
        tensor y_shuffled = shuffle(y_train, rd_state);

        tensor x_batch;
        tensor y_batch;

        tensor y_pred;
        tensor y_pred_val;

        for (auto j = 0; j < x_train.shape.front(); j += batch_size) {
            assert(0 < batch_size && batch_size <= x_train.shape.front());

            if (x_train.shape.front() <= j + batch_size) {
                x_batch = slice(x_shuffled, j, x_train.shape.front() - j);
                y_batch = slice(y_shuffled, j, x_train.shape.front() - j);
            } else {
                x_batch = slice(x_shuffled, j, batch_size);
                y_batch = slice(y_shuffled, j, batch_size);
            }

            std::vector<tensor> a = forward(x_batch, w_b.first, w_b.second);
            y_pred = a.back();

            std::vector<tensor> dl_dz, dl_dw, dl_db;

            for (auto k = lyrs.size() - 1; 0 < k; --k) {
                if (k == lyrs.size() - 1)
                    dl_dz.push_back(dl_da_da_dz(y_batch, y_pred));
                else
                    dl_dz.push_back(matmul(dl_dz[(lyrs.size() - 2) - k], transpose(w_b.first[k]), CPU) * da_dz(a[k - 1]));

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

        std::vector<tensor> a_val = forward(x_val, w_b.first, w_b.second);
        y_pred_val = a_val.back();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "Epoch " << i << "/" << epochs << std::endl;
        std::cout << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << loss(y_batch, y_pred) << " - accuracy: " << metric(y_batch, y_pred);
        std::cout << " - val_loss: " << loss(y_val, y_pred_val) << " - val_accuracy: " << metric(y_val, y_pred_val) << std::endl;
    }
}

float nn::evaluate(const tensor &x, const tensor &y) {
    std::vector<tensor> a = forward(x, w_b.first, w_b.second);
    return loss(y, a.back());
}

tensor nn::predict(const tensor &x) {
    std::vector<tensor> a = forward(x, w_b.first, w_b.second);
    return a.back();
}

std::pair<std::vector<tensor>, std::vector<tensor>> nn::init_params() {
    std::vector<tensor> w;
    std::vector<tensor> b;

    for (auto i = 0; i < lyrs.size() - 1; ++i) {
        w.push_back(normal_dist({lyrs[i], lyrs[i + 1]}, 0.0f, 0.2f));
        b.push_back(zeros({1, lyrs[i + 1]}));
    }

    return std::make_pair(w, b);
}

std::vector<tensor> nn::forward(const tensor &x, const std::vector<tensor> &w, const std::vector<tensor> &b) {
    std::vector<tensor> a;

    for (auto i = 0; i < lyrs.size() - 1; ++i) {
        if (i == 0) {
            // (64, 10) -> (64, 1) or (64, 10) I think latter is clearer, but
            // former is more performant. (10, 64) -> (1, 64) x.T = (4, 10), w1
            // = (64, 4), w2 = (10(must), 64), w3 = (64, 3), output = (64, 3) x
            // = (10, 4), w1 = (4, 64), w2 = (64, 64), w3 = (64, 3), ouput =
            // (10, 3)

            tensor z = matmul(x, w[i], CPU) + b[i];
            a.push_back(activations[i](z));
        } else {
            tensor z = matmul(a[i - 1], w[i], CPU) + b[i];
            a.push_back(activations[i](z));
        }
    }

    return a;
}

rnn::rnn(const act_func &activation, const loss_func &loss, const float lr) {
    this->activation = activation;
    this->loss = loss;
    this->lr = lr;

    h_t = zeros({hidden_size, batch_size});

    w_xh = uniform_dist({hidden_size, input_size});
    w_hh = uniform_dist({hidden_size, hidden_size});
    w_hy = uniform_dist({output_size, hidden_size});

    b_h = zeros({hidden_size, batch_size});
    b_y = zeros({output_size, batch_size});
}

void rnn::train(const tensor &x_train, const tensor &y_train, const tensor &x_val, const tensor &y_val) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto a = forward(x_train);
        auto y_pred = a.second.back();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        tensor dl_dy_pred = -2.0f / y_train.size * (transpose(y_train) - y_pred);
        tensor dl_dw_hy = matmul(dl_dy_pred, transpose(a.first.back()), CPU);
        tensor dl_dw_hh = zeros({hidden_size, hidden_size});
        tensor dl_db_h = zeros({hidden_size, batch_size});

        for (auto i = 0; i < seq_length; ++i) {
            tensor dy_pred_dh = w_hy;
            tensor dl_dh = matmul(transpose(dl_dy_pred), dy_pred_dh, CPU); // matmul(transpose(1 8316), 1 50)


            tensor dh_dw_hh = a.first.front() * (1.0f - activation(a.first[i]) * activation(a.first[i]));
            dl_dw_hh = dl_dw_hh + matmul(transpose(dl_dh), transpose(dh_dw_hh), CPU);

            dl_db_h = dl_db_h + transpose(dl_dh);
        }

        w_hh = w_hh - lr * dl_dw_hh;
        w_hy = w_hy - lr * dl_dw_hy;
        b_h = b_h - lr * dl_db_h;
        b_y = b_y - lr * dl_dy_pred; // dl_dy_pred should might be sum(dl_dy_pred, 0) which has shape (1, 8316) and sum(dl_dy_pred, 1) has (1, 1), but since dl_dy_pred is already (1, 8316) so...

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << loss(transpose(y_train), y_pred) << std::endl;
    }
}

std::pair<std::vector<tensor>, std::vector<tensor>> rnn::forward(const tensor &x) {
    std::vector<tensor> h;
    std::vector<tensor> y;

    for (auto i = 0; i < seq_length; ++i) {
        size_t idx = i;
        tensor x_t = zeros({batch_size - 10, input_size});

        // for (auto i = 0; i < batch_size * num_features; ++i)
        for (auto j = 0; j < batch_size - seq_length; ++j) {
            x_t[j] = x[idx];
            idx += seq_length;
        }

        if (i == seq_length - 1) {

            std::cout << x_t[0] << std::endl;
            std::cout << x_t[1] << std::endl;
            std::cout << x_t[2] << std::endl;
            std::cout << x_t[3] << std::endl;
        }

        // [0.00043548]
        // [0.00039868]
        // [0.00034961]
        // [0.00036495]
        // [0.00038335]
        // [0.00042322]
        // [0.00045695]
        // [0.00048762]
        // [0.00052749]
        // [0.00060109]
        // [0.00061336]
        // [0.00059189]
        // [0.00056736]
        // [0.00057656]
        // [0.00055816]
        // [0.00052135]

        // (now)
        // 50 1, 1 8316 = 50 8316 -> 50 50, 50 8316 = 50 8316 -> 1 50, 50 8316 = 1 8316
        // matmul(50 1, 1 8316) -> 50 8316 + matmul(50 50, 50 8316))) -> 50 8316
        // matmul(1 50, 50 8316) -> 1 8316

        // 8316 1, 1 50 = 8316 50 -> 8316 50, 50 50 = 8316 50 -> 8316 50, 50 1 = 8316 1

        // 50 8316, 8316 1 = 50 1 -> 50 50, 50 1 = 50 1 -> 1 50, 50 1 = 1 1
        // I think this is wrong because when you think about it it's weird that getting only one ouput even thougth I input 8316 batches.

        h_t = activation(matmul(w_xh, transpose(x_t), CPU) + matmul(w_hh, h_t, CPU) + b_h);
        tensor y_t = matmul(w_hy, h_t, CPU) + b_y; // I don't need to calculate this every steps if it is of type Many-to-one. Only calculate if it's of type like One-to-may, Many-to-many.


        h.push_back(h_t);
        y.push_back(y_t);
    }

    return std::make_pair(h, y);
}

tensor embedding(const size_t vocab_size, const size_t cols, const tensor &ind) {
    for (auto i = 0; i < ind.size; ++i)
        assert(ind[i] < vocab_size);

    tensor embeddings_mat = uniform_dist({vocab_size, cols});

    std::cout << embeddings_mat << std::endl;

    tensor dense_vecs = zeros({ind.shape.front(), ind.shape.back(), cols});

    for (auto i = 0; i < ind.size; ++i) {
        auto a = slice(embeddings_mat, ind[i], 1);

        for (auto j = 0; j < a.size; ++j)
            dense_vecs[cols * i + j] = a[j];
    }

    return dense_vecs;
}