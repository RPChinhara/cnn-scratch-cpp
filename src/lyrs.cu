#include "lyrs.h"
#include "arrs.h"
#include "math.hpp"
#include "preproc.h"
#include "rd.h"
#include "tensor.h"

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

    cudaMemcpy(d_A, t1.elem, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, t2.elem, N * P * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((P + threadsPerBlock.x - 1) / threadsPerBlock.x,(M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, P);

    cudaMemcpy(t_new.elem, d_C, M * P * sizeof(float), cudaMemcpyDeviceToHost);

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
   return std::vector<tensor>();
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

void matmul_cpu() {

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
                    dl_dz.push_back(matmul(dl_dz[(lyrs.size() - 2) - k], transpose(w_b.first[k])) * da_dz(a[k - 1]));

                if (k == 1)
                    dl_dw.push_back(matmul(transpose(x_batch), dl_dz[(lyrs.size() - 1) - k]));
                else
                    dl_dw.push_back(matmul(transpose(a[k - 2]), dl_dz[(lyrs.size() - 1) - k]));

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
            tensor z = matmul(x, w[i]) + b[i];
            a.push_back(activations[i](z));
        } else {
            tensor z = matmul(a[i - 1], w[i]) + b[i];
            a.push_back(activations[i](z));
        }
    }

    return a;
}

rnn::rnn(const act_func &activation, const loss_func &loss, const float lr) {
    this->activation = activation;
    this->loss = loss;
    this->lr = lr;

    w_xh = uniform_dist({hidden_size, input_size});
    w_hh = uniform_dist({hidden_size, hidden_size});
    w_hy = uniform_dist({output_size, hidden_size});

    b_h = zeros({hidden_size, batch_size});
    b_y = zeros({output_size, batch_size});
}

void rnn::train(const tensor &x_train, const tensor &y_train, const tensor &x_val, const tensor &y_val) {

    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto h_y = forward(x_train);
        auto y = h_y.second.back();

        tensor d_loss_d_w_xh = zeros({hidden_size, input_size});
        tensor d_loss_d_w_hh = zeros({hidden_size, hidden_size});
        tensor d_loss_d_b_h  = zeros({hidden_size, batch_size});

        float num_samples = static_cast<float>(y_train.shape.front());

        tensor d_loss_d_y = -2.0f / num_samples * (transpose(y_train) - y);

        tensor d_loss_d_w_hy  = matmul(d_loss_d_y, transpose(h_y.first.back()));

        // d_loss_d_h_10                   -> (8317, 50)
        // 1.0f - square(h_y.first.back()) -> (50, 8317)
        // h_y.first[h_y.first.size() - 1] -> (50, 8317)
        // h_y.first.size()                -> 11
        // d_loss_d_w_hh_10                -> (50, 50)
        // d_loss_d_h_t_9                  -> (8317, 50)

        // for (auto j = 0; j < seq_length; ++j) {
            // tensor d_y_d_h_10 = w_hy;
            // tensor d_loss_d_h_10 = matmul(transpose(d_loss_d_y), d_y_d_h_10);
            // tensor d_loss_d_w_hh_10 = matmul((transpose(d_loss_d_h_10) * (1.0f - square(h_y.first[10]))), transpose(h_y.first[9]));

            // tensor d_loss_d_h_t_9  = matmul(d_loss_d_h_10 * transpose(1.0f - square(h_y.first[10])), w_hh);
            // tensor d_loss_d_w_hh_9 = matmul((transpose(d_loss_d_h_t_9) * (1.0f - square(h_y.first[9]))), transpose(h_y.first[8]));

            // tensor d_loss_d_h_t_8  = matmul(d_loss_d_h_t_9 * transpose(1.0f - square(h_y.first[9])), w_hh);
            // tensor d_loss_d_w_hh_8 = matmul((transpose(d_loss_d_h_t_8) * (1.0f - square(h_y.first[8]))), transpose(h_y.first[7]));

            // tensor d_loss_d_h_t_7  = matmul(d_loss_d_h_t_8 * transpose(1.0f - square(h_y.first[8])), w_hh);
            // tensor d_loss_d_w_hh_7 = matmul((transpose(d_loss_d_h_t_7) * (1.0f - square(h_y.first[7]))), transpose(h_y.first[6]));

            // tensor d_loss_d_h_t_6  = matmul(d_loss_d_h_t_7 * transpose(1.0f - square(h_y.first[7])), w_hh);
            // tensor d_loss_d_w_hh_6 = matmul((transpose(d_loss_d_h_t_6) * (1.0f - square(h_y.first[6]))), transpose(h_y.first[5]));

            // tensor d_loss_d_h_t_5  = matmul(d_loss_d_h_t_6 * transpose(1.0f - square(h_y.first[6])), w_hh);
            // tensor d_loss_d_w_hh_5 = matmul((transpose(d_loss_d_h_t_5) * (1.0f - square(h_y.first[5]))), transpose(h_y.first[4]));

            // d_loss_d_w_hh = d_loss_d_w_hh + d_loss_d_w_hh_9 + d_loss_d_w_hh_8 + d_loss_d_w_hh_7 + d_loss_d_w_hh_6 + d_loss_d_w_hh_5;

            // tensor d_loss_d_w_xh_10
            // tensor d_loss_d_w_xh_9
            // tensor d_loss_d_w_xh_8
            // tensor d_loss_d_w_xh_7

            // NOTE: I think I have to add all the dL/dht e.g., dL2/dh2 + dL2/dh1
            // d_loss_d_b_h = d_loss_d_b_h + transpose(d_loss_d_h_10);
        // }

        tensor d_loss_d_h_t = zeros({batch_size, hidden_size});
        // tensor d_loss_d_h_t;
        for (auto j = seq_length; j > 0; --j) {
            if (j == seq_length) {
                tensor d_y_d_h_10 = w_hy;
                d_loss_d_h_t = matmul(transpose(d_loss_d_y), d_y_d_h_10);
            } else {
                d_loss_d_h_t = matmul(d_loss_d_h_t * transpose(1.0f - square(h_y.first[j + 1])), w_hh);
            }

            d_loss_d_w_hh = d_loss_d_w_hh + matmul((transpose(d_loss_d_h_t) * (1.0f - square(h_y.first[j]))), transpose(h_y.first[j - 1]));
        }

        // NOTE: These a = a - lr * b is working. I've checked.
        w_xh = w_xh - lr * d_loss_d_w_xh; // incomplete
        w_hh = w_hh - lr * d_loss_d_w_hh; // done
        w_hy = w_hy - lr * d_loss_d_w_hy; // done

        b_h = b_h - lr * d_loss_d_b_h;    // incomplete
        b_y = b_y - lr * d_loss_d_y;      // done

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << loss(transpose(y_train), y) << std::endl;
    }
}

float rnn::evaluate(const tensor &x, const tensor &y) {
    return 0.0f;
}

tensor rnn::predict(const tensor &x) {
    return tensor();
}

std::pair<std::vector<tensor>, std::vector<tensor>> rnn::forward(const tensor &x) {
    std::vector<tensor> h;
    std::vector<tensor> y;

    h_t = zeros({hidden_size, batch_size});
    h.push_back(h_t);

    for (auto i = 0; i < seq_length; ++i) {
        size_t idx = i;

        tensor x_t = zeros({batch_size, input_size});

        for (auto j = 0; j < batch_size; ++j) {
            x_t[j] = x[idx];
            idx += seq_length;
        }

        h_t = activation(matmul(w_xh, transpose(x_t)) + matmul(w_hh, h_t) + b_h);
        tensor y_t = matmul(w_hy, h_t) + b_y;

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