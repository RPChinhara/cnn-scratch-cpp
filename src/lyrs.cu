#include "lyrs.h"
#include "acts.h"
#include "arrs.h"
#include "math.hpp"
#include "preproc.h"
#include "rd.h"
#include "tensor.h"

#include <array>
#include <cassert>
#include <chrono>

__global__ void matmul(float* a, float* b, float* c, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float value = 0;
        for (int k = 0; k < n; ++k) {
            value += a[row * n + k] * b[k * p + col];
        }

        c[row * p + col] = value;
    }
}

tensor matmul(const tensor &t1, const tensor &t2) {
    assert(t1.shape.back() == t2.shape.front());

    tensor t_new = zeros({t1.shape.front(), t2.shape.back()});

    int M = t1.shape.front();
    int N = t1.shape.back();
    int P = t2.shape.back();

    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * P * sizeof(float));
    cudaMalloc(&d_C, M * P * sizeof(float));

    cudaMemcpy(d_A, t1.elems, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, t2.elems, N * P * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x,(M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, P);

    cudaMemcpy(t_new.elems, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return t_new;
}

static size_t get_batch_size(const std::vector<size_t> &shape) {
    assert(1 < shape.size());
    size_t batchSize = 1;

    for (auto i = 0; i < shape.size() - 2; ++i)
        batchSize *= shape[i];

    return batchSize;
}

tensor transpose(const tensor &t) {
    assert(2 <= t.shape.size());

    tensor t_new = zeros({t.shape.back(), t.shape[t.shape.size() - 2]});

    std::vector<size_t> idx_rows;

    for (auto i = 0; i < t.size; ++i)
        idx_rows.push_back(i * t.shape.back());

    size_t batchSize = get_batch_size(t.shape);

    size_t idx = 0;

    for (auto i = 0; i < batchSize; ++i) {
        for (auto j = 0; j < t_new.shape[t_new.shape.size() - 2]; ++j) {
            for (auto k = 0; k < t_new.shape.back(); ++k) {
                t_new[idx] = t[idx_rows[k + (i * t_new.shape.back())]];
                idx_rows[k + (i * t_new.shape.back())] += 1;
                ++idx;
            }
        }
    }

    return t_new;
}

cnn2d::cnn2d(const std::vector<size_t> &filters, float const lr) {
    this->filters = filters;
    this->lr = lr;
}

void cnn2d::train(const tensor &xTrain, const tensor &yTrain, const tensor &xVal, const tensor &yVal) {
    tensor kernel = tensor({3, 3}, {1, -1, 1, 0, 1, 0, -1, 0, 1});

    size_t kernelHeight = kernel.shape.front();
    size_t kernelWidth = kernel.shape.back();

    size_t inputHeight = xTrain.shape[1];
    size_t inputWidth = xTrain.shape[2];

    size_t outputHeight = inputHeight - kernelHeight + 1;
    size_t outputWidth = inputWidth - kernelWidth + 1;

    tensor output = zeros({outputHeight, outputWidth});
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

    std::vector<tensor> w;
    std::vector<tensor> b;

    return std::make_pair(w, b);
}

std::vector<tensor> gru::forward(const tensor &x) {
    init_params();

    return std::vector<tensor>();
}

lstm::lstm(const loss_func &loss, const float lr) {
    this->loss = loss;
    this->lr = lr;

    w_f = glorot_uniform(hidden_size, hidden_size + input_size);
    w_i = glorot_uniform(hidden_size, hidden_size + input_size);
    w_c = glorot_uniform(hidden_size, hidden_size + input_size);
    w_o = glorot_uniform(hidden_size, hidden_size + input_size);
    w_y = glorot_uniform(output_size, hidden_size);

    b_f = zeros({hidden_size, 1});
    b_i = zeros({hidden_size, 1});
    b_c = zeros({hidden_size, 1});
    b_o = zeros({hidden_size, 1});
    b_y = zeros({output_size, 1});
}

void lstm::train(const tensor &x_train, const tensor &y_train) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto [x_sequence, concat_sequence, z_f_sequence, z_i_sequence, i_sequence, z_c_tilde_sequence, c_tilde_sequence, c_sequence, z_o_sequence, o_sequence, h_sequence, y_sequence] = forward(x_train, Phase::TRAIN);

        float error = loss(transpose(y_train), y_sequence.front());

        tensor d_loss_d_h_t_w_f = zeros({batch_size, hidden_size});
        tensor d_loss_d_h_t_w_i = zeros({batch_size, hidden_size});
        tensor d_loss_d_h_t_w_c = zeros({batch_size, hidden_size});
        tensor d_loss_d_h_t_w_o = zeros({batch_size, hidden_size});

        tensor d_loss_d_w_f = zeros({hidden_size, hidden_size + input_size});
        tensor d_loss_d_w_i = zeros({hidden_size, hidden_size + input_size});
        tensor d_loss_d_w_c = zeros({hidden_size, hidden_size + input_size});
        tensor d_loss_d_w_o = zeros({hidden_size, hidden_size + input_size});

        tensor d_loss_d_b_f  = zeros({hidden_size, 1});
        tensor d_loss_d_b_i  = zeros({hidden_size, 1});
        tensor d_loss_d_b_c  = zeros({hidden_size, 1});
        tensor d_loss_d_b_o  = zeros({hidden_size, 1});

        float num_samples = static_cast<float>(y_train.shape.front());
        tensor d_loss_d_y = -2.0f / num_samples * (transpose(y_train) - y_sequence.front());

        // c_sequence.size()       11
        // o_sequence.size()       10
        // z_o_sequence.size()     10
        // concat_sequence.size()  10
        // c_tilde_sequence.size() 10

        // (1.0f - square(hyperbolic_tangent(c_sequence[0]))) 50 8317
        // o_sequence[0]                                      50 8317

        for (auto j = seq_length; j > 0; --j) {
            if (j == seq_length) {
                tensor d_y_d_h_10 = w_y;

                d_loss_d_h_t_w_f = matmul(transpose(d_loss_d_y), d_y_d_h_10);
                d_loss_d_h_t_w_i = matmul(transpose(d_loss_d_y), d_y_d_h_10);
                d_loss_d_h_t_w_c = matmul(transpose(d_loss_d_y), d_y_d_h_10);
                d_loss_d_h_t_w_o = matmul(transpose(d_loss_d_y), d_y_d_h_10);
            } else {
                d_loss_d_h_t_w_f = matmul(d_loss_d_h_t_w_f * transpose(o_sequence[j] * (1.0f - square(hyperbolic_tangent(c_sequence[j + 1]))) * c_sequence[j + 1] * sigmoid_derivative(z_f_sequence[j])), vslice(w_f, w_f.shape.back() - 1));
                d_loss_d_h_t_w_i = matmul(d_loss_d_h_t_w_i * transpose(o_sequence[j] * (1.0f - square(hyperbolic_tangent(c_sequence[j + 1]))) * c_tilde_sequence[j] * sigmoid_derivative(z_i_sequence[j])), vslice(w_i, w_i.shape.back() - 1));
                d_loss_d_h_t_w_c = matmul(d_loss_d_h_t_w_c * transpose(o_sequence[j] * (1.0f - square(hyperbolic_tangent(c_sequence[j + 1]))) * i_sequence[j] * (1.0f - square(hyperbolic_tangent(z_c_tilde_sequence[j])))), vslice(w_c, w_c.shape.back() - 1));
                d_loss_d_h_t_w_o = matmul(d_loss_d_h_t_w_o * transpose(hyperbolic_tangent(c_sequence[j + 1]) * sigmoid_derivative(z_o_sequence[j])), vslice(w_o, w_o.shape.back() - 1));
                                       // 8317, 50                     50, 8317                                50, 8317                              50, 50
            }

            d_loss_d_w_f = d_loss_d_w_f + matmul(transpose(d_loss_d_h_t_w_f) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * c_sequence[j] * sigmoid_derivative(z_f_sequence[j - 1]), transpose(concat_sequence[j - 1]));
            d_loss_d_w_i = d_loss_d_w_i + matmul(transpose(d_loss_d_h_t_w_i) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * c_tilde_sequence[j - 1] * sigmoid_derivative(z_i_sequence[j - 1]), transpose(concat_sequence[j - 1]));
            d_loss_d_w_c = d_loss_d_w_c + matmul(transpose(d_loss_d_h_t_w_c) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * i_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(z_c_tilde_sequence[j - 1]))), transpose(concat_sequence[j - 1]));
            d_loss_d_w_o = d_loss_d_w_o + matmul(transpose(d_loss_d_h_t_w_o) * hyperbolic_tangent(c_sequence[j]) * sigmoid_derivative(z_o_sequence[j - 1]), transpose(concat_sequence[j - 1]));
                                                        // 8317, 50            50, 8317                            50, 8317                                 51, 8317

            d_loss_d_b_f = d_loss_d_b_f + sum(transpose(d_loss_d_h_t_w_f) * sigmoid_derivative(z_f_sequence[j - 1]), 1);
            d_loss_d_b_i = d_loss_d_b_i + sum(transpose(d_loss_d_h_t_w_i) * sigmoid_derivative(z_i_sequence[j - 1]), 1);
            d_loss_d_b_c = d_loss_d_b_c + sum(transpose(d_loss_d_h_t_w_c) * (1.0f - square(hyperbolic_tangent(z_c_tilde_sequence[j - 1]))), 1);
            d_loss_d_b_o = d_loss_d_b_o + sum(transpose(d_loss_d_h_t_w_o) * sigmoid_derivative(z_o_sequence[j - 1]), 1);
        }

        // tensor concat = vstack({h_t, transpose(x_t)});

        // tensor z_f = matmul(w_f, concat) + b_f;
        // tensor f_t = sigmoid(z_f);

        // tensor z_i = matmul(w_i, concat) + b_i;
        // tensor i_t = sigmoid(z_i);

        // tensor z_c_tilde_t = matmul(w_c, concat) + b_c;
        // tensor c_tilde_t = hyperbolic_tangent(z_c_tilde_t);

        // c_t = f_t * c_t + i_t * c_tilde_t;

        // tensor z_o = matmul(w_o, concat) + b_o;
        // tensor o_t = sigmoid(z_o);

        // h_t = o_t * hyperbolic_tangent(c_t);

        // tensor y_t = matmul(w_y, h_t) + b_y;

        // -------------------------------------------------------------------------------------------------------------------------
        // Check forward pass on https://en.wikipedia.org/wiki/Long_short-term_memory to remind myself that way to compute gradients make sense.
        // (dL/dy * dy/dh_10) * dh_10/do_10 * do_t10/dw_o
        // (dL/dy * dy/dh_10 * dh_10/do_10 * do_10/dh_9) * dh_9/do_9 * do_9/dw_o
        // (dL/dy * dy/dh_10 * dh_10/do_10 * do_10/dh_9 * dh_9/do_9 * do_9/dh_8) * dh_8/do_8 * do_8/dw_o
        // -------------------------------------------------------------------------------------------------------------------------
        // (dL/dy * dy/dh_10) * dh_10/dc_10 * dc_10/dc_tilde_10 * dc_tilde_10/dw_c
        // (dL/dy * dy/dh_10 * dh_10/dc_10 * dc_10/dc_tilde_10 * dc_tilde_10/dh_9) * dh_9/dc_9 * dc_9/dc_tilde_9 * dc_tilde9/dw_c

        // dh10/do10 * do10/wo
        // dh10/do10 * do10/dh9 * dh9/do9 * do9/wo

        // dh10/dc10 * dc10/d~c10 * d~c10/w_c
        // dh10/dc10 * dc10/d~c10 * d~c10/dh9 * dh9/dc9 * dc9/d~c9 * d~c9/wc
        // -------------------------------------------------------------------------------------------------------------------------
        // (dL/dy * dy/dh_10) * dh_10/dc_10 * dc_10/di_10 * di_10/dw_i
        // (dL/dy * dy/dh_10) * dh_10/dc_10 * dc_10/di_10 * di_10/dh_9) * dh_9/dc_9 * dc_9/di_9 * di_9/dw_i
        // -------------------------------------------------------------------------------------------------------------------------
        // (dL/dy * dy/dh_10) * dh_10/dc_10 * dc_10/df_10 * df_10/dw_f
        // (dL/dy * dy/dh_10) * dh_10/dc_10 * dc_10/df_10 * df_10/dh_9) * dh_9/dc_9 * dc_9/df_9 * df_9/dw_f
        // -------------------------------------------------------------------------------------------------------------------------

        tensor d_loss_d_w_y  = matmul(d_loss_d_y, transpose(h_sequence.back()));

        w_f = w_f - lr * d_loss_d_w_f;
        w_i = w_i - lr * d_loss_d_w_i;
        w_c = w_c - lr * d_loss_d_w_c;
        w_o = w_o - lr * d_loss_d_w_o;
        w_y = w_y - lr * d_loss_d_w_y;

        b_f = b_f - lr * d_loss_d_b_f;
        b_i = b_i - lr * d_loss_d_b_i;
        b_c = b_c - lr * d_loss_d_b_c;
        b_o = b_o - lr * d_loss_d_b_o;
        b_y = b_y - lr * d_loss_d_y;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << error << std::endl;
    }
}

float lstm::evaluate(const tensor &x, const tensor &y) {
    auto [x_sequence, concat_sequence, z_f_sequence, z_i_sequence, i_sequence, z_c_tilde_sequence, c_tilde_sequence, c_sequence, z_o_sequence, o_sequence, h_sequence, y_sequence] = forward(x, Phase::TEST);
    return loss(transpose(y), y_sequence.front());
}

tensor lstm::predict(const tensor &x) {
    auto [x_sequence, concat_sequence, z_f_sequence, z_i_sequence, i_sequence, z_c_tilde_sequence, c_tilde_sequence, c_sequence, z_o_sequence, o_sequence, h_sequence, y_sequence] = forward(x, Phase::TEST);
    return transpose(y_sequence.front());
}

std::array<std::vector<tensor>, 12> lstm::forward(const tensor &x, enum Phase phase) {
    std::vector<tensor> x_sequence;
    std::vector<tensor> concat_sequence;
    std::vector<tensor> z_f_sequence;
    std::vector<tensor> z_i_sequence;
    std::vector<tensor> i_sequence;
    std::vector<tensor> z_c_tilde_sequence;
    std::vector<tensor> c_tilde_sequence;
    std::vector<tensor> c_sequence;
    std::vector<tensor> z_o_sequence;
    std::vector<tensor> o_sequence;
    std::vector<tensor> h_sequence;
    std::vector<tensor> y_sequence;

    if (phase == Phase::TRAIN)
        batch_size = 8317;
    else
        batch_size = 2072;

    tensor c_t = zeros({hidden_size, batch_size});
    c_sequence.push_back(c_t);

    tensor h_t = zeros({hidden_size, batch_size});
    h_sequence.push_back(h_t);

    for (auto i = 0; i < seq_length; ++i) {
        size_t idx = i;

        tensor x_t = zeros({batch_size, input_size});

        for (auto j = 0; j < batch_size; ++j) {
            x_t[j] = x[idx];
            idx += seq_length;
        }

        tensor concat_t = vstack({h_t, transpose(x_t)});

        tensor z_f_t = matmul(w_f, concat_t) + b_f;
        tensor f_t = sigmoid(z_f_t);

        tensor z_i_t = matmul(w_i, concat_t) + b_i;
        tensor i_t = sigmoid(z_i_t);

        tensor z_c_tilde_t = matmul(w_c, concat_t) + b_c;
        tensor c_tilde_t = hyperbolic_tangent(z_c_tilde_t);

        c_t = f_t * c_t + i_t * c_tilde_t;

        tensor z_o_t = matmul(w_o, concat_t) + b_o;
        tensor o_t = sigmoid(z_o_t);

        h_t = o_t * hyperbolic_tangent(c_t);

        tensor y_t = matmul(w_y, h_t) + b_y;

        x_sequence.push_back(x_t);
        concat_sequence.push_back(concat_t);
        z_f_sequence.push_back(z_f_t);
        z_i_sequence.push_back(z_i_t);
        i_sequence.push_back(z_i_t);
        z_c_tilde_sequence.push_back(z_c_tilde_t);
        c_tilde_sequence.push_back(c_tilde_t);
        c_sequence.push_back(c_t);
        z_o_sequence.push_back(z_o_t);
        o_sequence.push_back(o_t);
        h_sequence.push_back(h_t);

        if (i == seq_length - 1)
            y_sequence.push_back(y_t);

        // std::cout << concat.shape.front() << " " << concat.shape.back() << std::endl;
        // std::cout << f_t.shape.front() << " " << f_t.shape.back() << std::endl;
        // std::cout << i_t.shape.front() << " " << i_t.shape.back() << std::endl;
        // std::cout << c_tilde_t.shape.front() << " " << c_tilde_t.shape.back() << std::endl;
        // std::cout << c_t.shape.front() << " " << c_t.shape.back() << std::endl;
        // std::cout << o_t.shape.front() << " " << o_t.shape.back() << std::endl;
        // std::cout << h_t.shape.front() << " " << h_t.shape.back() << std::endl;
        // std::cout << y_t.shape.front() << " " << y_t.shape.back() << std::endl;
    }

    std::array<std::vector<tensor>, 12> sequences;

    sequences[0]  = x_sequence;
    sequences[1]  = concat_sequence;
    sequences[2]  = z_f_sequence;
    sequences[3]  = z_i_sequence;
    sequences[4]  = i_sequence;
    sequences[5]  = z_c_tilde_sequence;
    sequences[6]  = c_tilde_sequence;
    sequences[7]  = c_sequence;
    sequences[8]  = z_o_sequence;
    sequences[9]  = o_sequence;
    sequences[10]  = h_sequence;
    sequences[11] = y_sequence;

    return sequences;
}

nn::nn(const std::vector<size_t> &lyrs, const std::vector<act_func> &activations, const loss_func &loss, const metric_func &metric, const float lr) {
    this->lyrs = lyrs;
    this->activations = activations;
    this->loss = loss;
    this->metric = metric;
    this->lr = lr;

    w_b = init_params();
    w_b_momentum = init_params();
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

        float accumulated_loss = 0.0f;

        for (auto j = 0; j < x_train.shape.front(); j += batch_size) {
            assert(0 < batch_size && batch_size <= x_train.shape.front());

            if (x_train.shape.front() <= j + batch_size) {
                x_batch = slice(x_shuffled, j, x_train.shape.front() - j);
                y_batch = slice(y_shuffled, j, x_train.shape.front() - j);
            } else {
                x_batch = slice(x_shuffled, j, batch_size);
                y_batch = slice(y_shuffled, j, batch_size);
            }

            auto [z, a] = forward(x_batch, w_b.first, w_b.second);
            y_pred = a.back();

            accumulated_loss += loss(y_batch, y_pred);

            std::vector<tensor> dl_dz, dl_dw, dl_db;

            for (auto k = lyrs.size() - 1; 0 < k; --k) {
                if (k == lyrs.size() - 1)
                    dl_dz.push_back(y_pred - y_batch);
                else
                    dl_dz.push_back(matmul(dl_dz[(lyrs.size() - 2) - k], transpose(w_b.first[k])) * relu_derivative(z[k - 1]));

                if (k == 1)
                    dl_dw.push_back(matmul(transpose(x_batch), dl_dz[(lyrs.size() - 1) - k]));
                else
                    dl_dw.push_back(matmul(transpose(a[k - 2]), dl_dz[(lyrs.size() - 1) - k]));

                dl_db.push_back(sum(dl_dz[(lyrs.size() - 1) - k], 0));

                dl_dw[(lyrs.size() - 1) - k] = clip_by_value(dl_dw[(lyrs.size() - 1) - k], -8.0f, 8.0f);
                dl_db[(lyrs.size() - 1) - k] = clip_by_value(dl_db[(lyrs.size() - 1) - k], -8.0f, 8.0f);

                w_b_momentum.first[k - 1] = momentum * w_b_momentum.first[k - 1] - lr * dl_dw[(lyrs.size() - 1) - k];
                w_b_momentum.second[k - 1] = momentum * w_b_momentum.second[k - 1] - lr * dl_db[(lyrs.size() - 1) - k];

                w_b.first[k - 1] += w_b_momentum.first[k - 1];
                w_b.second[k - 1] += w_b_momentum.second[k - 1];
            }

            dl_dz.clear(), dl_dw.clear(), dl_db.clear();
        }

        auto [z_val, a_val] = forward(x_val, w_b.first, w_b.second);
        y_pred_val = a_val.back();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << std::fixed << std::setprecision(5);
        std::cout << "Epoch " << i << "/" << epochs << std::endl;
        std::cout << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << accumulated_loss / ceil(x_train.shape.front() / batch_size) << " - accuracy: " << metric(y_batch, y_pred);
        std::cout << " - val_loss: " << loss(y_val, y_pred_val) << " - val_accuracy: " << metric(y_val, y_pred_val) << std::endl;
    }
}

float nn::evaluate(const tensor &x, const tensor &y) {
    auto [z, a] = forward(x, w_b.first, w_b.second);
    return loss(y, a.back());
}

tensor nn::predict(const tensor &x) {
    auto [z, a] = forward(x, w_b.first, w_b.second);
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

std::pair<std::vector<tensor>, std::vector<tensor>>  nn::forward(const tensor &x, const std::vector<tensor> &w, const std::vector<tensor> &b) {
    std::vector<tensor> zs;
    std::vector<tensor> as;

    for (auto i = 0; i < lyrs.size() - 1; ++i) {
        if (i == 0) {
            tensor z = matmul(x, w[i]) + b[i];
            tensor a = activations[i](z);

            zs.push_back(z);
            as.push_back(a);
        } else {
            tensor z = matmul(as[i - 1], w[i]) + b[i];
            tensor a = activations[i](z);

            zs.push_back(z);
            as.push_back(a);
        }
    }

    return std::make_pair(zs, as);
}

rnn::rnn(const act_func &activation, const loss_func &loss, const float lr) {
    this->activation = activation;
    this->loss = loss;
    this->lr = lr;

    w_xh = glorot_uniform(hidden_size, input_size);
    w_hh = glorot_uniform(hidden_size, hidden_size);
    w_hy = glorot_uniform(output_size, hidden_size);

    b_h = zeros({hidden_size, 1});
    b_y = zeros({output_size, 1});

    m_w_xh = zeros({hidden_size, input_size});
    m_w_hh = zeros({hidden_size, hidden_size});
    m_w_hy = zeros({output_size, hidden_size});
    m_b_h  = zeros({hidden_size, 1});
    m_b_y  = zeros({output_size, 1});

    v_w_xh = zeros({hidden_size, input_size});
    v_w_hh = zeros({hidden_size, hidden_size});
    v_w_hy = zeros({output_size, hidden_size});
    v_b_h  = zeros({hidden_size, 1});
    v_b_y  = zeros({output_size, 1});
}

void rnn::train(const tensor &x_train, const tensor &y_train) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto [x_sequence, z_sequence, h_sequence, y_sequence] = forward(x_train, Phase::TRAIN);

        float error = loss(transpose(y_train), y_sequence.front());

        tensor d_loss_d_h_t = zeros({batch_size, hidden_size});
        tensor d_loss_d_w_xh = zeros({hidden_size, input_size});
        tensor d_loss_d_w_hh = zeros({hidden_size, hidden_size});
        tensor d_loss_d_b_h  = zeros({hidden_size, 1});

        float num_samples = static_cast<float>(y_train.shape.front());
        tensor d_loss_d_y = -2.0f / num_samples * (transpose(y_train) - y_sequence.front());

        for (auto j = seq_length; j > 0; --j) {
            if (j == seq_length) {
                tensor d_y_d_h_10 = w_hy;
                d_loss_d_h_t = matmul(transpose(d_loss_d_y), d_y_d_h_10);
            } else {
                d_loss_d_h_t = matmul(d_loss_d_h_t * transpose(relu_derivative(z_sequence[j])), w_hh);
            }

            d_loss_d_w_xh = d_loss_d_w_xh + matmul((transpose(d_loss_d_h_t) * relu_derivative(z_sequence[j - 1])), x_sequence[j - 1]);
            d_loss_d_w_hh = d_loss_d_w_hh + matmul((transpose(d_loss_d_h_t) * relu_derivative(z_sequence[j - 1])), transpose(h_sequence[j - 1]));

            d_loss_d_b_h  = d_loss_d_b_h + sum(transpose(d_loss_d_h_t) * relu_derivative(z_sequence[j - 1]), 1);
        }

        tensor d_loss_d_w_hy  = matmul(d_loss_d_y, transpose(h_sequence.back()));

        t += 1;

        m_w_xh = beta1 * m_w_xh + (1.0f - beta1) * d_loss_d_w_xh;
        m_w_hh = beta1 * m_w_hh + (1.0f - beta1) * d_loss_d_w_hh;
        m_w_hy = beta1 * m_w_hy + (1.0f - beta1) * d_loss_d_w_hy;
        m_b_h = beta1 * m_b_h + (1.0f - beta1) * d_loss_d_b_h;
        m_b_y = beta1 * m_b_y + (1.0f - beta1) * d_loss_d_y;

        v_w_xh = beta2 * v_w_xh + (1.0f - beta2) * square(d_loss_d_w_xh);
        v_w_hh = beta2 * v_w_hh + (1.0f - beta2) * square(d_loss_d_w_hh);
        v_w_hy = beta2 * v_w_hy + (1.0f - beta2) * square(d_loss_d_w_hy);
        v_b_h = beta2 * v_b_h + (1.0f - beta2) * square(d_loss_d_b_h);
        v_b_y = beta2 * v_b_y + (1.0f - beta2) * square(d_loss_d_y);

        tensor m_hat_w_xh = m_w_xh / (1.0f - powf(beta1, t));
        tensor m_hat_w_hh = m_w_hh / (1.0f - powf(beta1, t));
        tensor m_hat_w_hy = m_w_hy / (1.0f - powf(beta1, t));
        tensor m_hat_b_h = m_b_h / (1.0f - powf(beta1, t));
        tensor m_hat_b_y = m_b_y / (1.0f - powf(beta1, t));

        tensor v_hat_w_xh = v_w_xh / (1.0f - powf(beta2, t));
        tensor v_hat_w_hh = v_w_hh / (1.0f - powf(beta2, t));
        tensor v_hat_w_hy = v_w_hy / (1.0f - powf(beta2, t));
        tensor v_hat_b_h = v_b_h / (1.0f - powf(beta2, t));
        tensor v_hat_b_y = v_b_y / (1.0f - powf(beta2, t));

        w_xh = w_xh - lr * m_hat_w_xh / (sqrt(v_hat_w_xh) + epsilon);
        w_hh = w_hh - lr * m_hat_w_hh / (sqrt(v_hat_w_hh) + epsilon);
        w_hy = w_hy - lr * m_hat_w_hy / (sqrt(v_hat_w_hy) + epsilon);
        b_h = b_h - lr * m_hat_b_h / (sqrt(v_hat_b_h) + epsilon);
        b_y = b_y - lr * m_hat_b_y / (sqrt(v_hat_b_y) + epsilon);

        // w_xh = w_xh - lr * d_loss_d_w_xh;
        // w_hh = w_hh - lr * d_loss_d_w_hh;
        // w_hy = w_hy - lr * d_loss_d_w_hy;

        // b_h = b_h - lr * d_loss_d_b_h;
        // b_y = b_y - lr * d_loss_d_y;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << error << std::endl;
    }
}

float rnn::evaluate(const tensor &x, const tensor &y) {
    auto [x_sequence, z_sequence, h_sequence, y_sequence] = forward(x, Phase::TEST);
    return loss(transpose(y), y_sequence.front());
}

tensor rnn::predict(const tensor &x) {
    auto [x_sequence, z_sequence, h_sequence, y_sequence] = forward(x, Phase::TEST);
    return transpose(y_sequence.front());
}

std::tuple<std::vector<tensor>, std::vector<tensor>, std::vector<tensor>, std::vector<tensor>> rnn::forward(const tensor &x, enum Phase phase) {
    std::vector<tensor> x_sequence;
    std::vector<tensor> z_sequence;
    std::vector<tensor> h_sequence;
    std::vector<tensor> y_sequence;

    if (phase == Phase::TRAIN)
        batch_size = 8317;
    else
        batch_size = 2072;

    tensor h_t = zeros({hidden_size, batch_size});
    h_sequence.push_back(h_t);

    for (auto i = 0; i < seq_length; ++i) {
        size_t idx = i;

        tensor x_t = zeros({batch_size, input_size});

        for (auto j = 0; j < batch_size; ++j) {
            x_t[j] = x[idx];
            idx += seq_length;
        }

        tensor z_t = matmul(w_xh, transpose(x_t)) + matmul(w_hh, h_t) + b_h;
        h_t = activation(z_t);
        tensor y_t = matmul(w_hy, h_t) + b_y;

        x_sequence.push_back(x_t);
        z_sequence.push_back(z_t);
        h_sequence.push_back(h_t);

        if (i == seq_length - 1)
            y_sequence.push_back(y_t);
    }

    return std::make_tuple(x_sequence, z_sequence, h_sequence, y_sequence);
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