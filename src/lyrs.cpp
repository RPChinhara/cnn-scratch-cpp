#include "lyrs.h"
#include "acts.h"
#include "arrs.h"
#include "linalg.h"
#include "losses.h"
#include "math.hpp"
#include "preproc.h"
#include "rd.h"
#include "tensor.h"

#include <cassert>
#include <chrono>

gru::gru(const float lr, const size_t vocab_size) {
    this->lr = lr;
    this->vocab_size = vocab_size;

    w_z = glorot_uniform(hidden_size, hidden_size + input_size);
    w_r = glorot_uniform(hidden_size, hidden_size + input_size);
    w_h = glorot_uniform(hidden_size, hidden_size + input_size);
    w_y = glorot_uniform(output_size, hidden_size);

    b_z = zeros({hidden_size, 1});
    b_r = zeros({hidden_size, 1});
    b_h = zeros({hidden_size, 1});
    b_y = zeros({output_size, 1});

    m_w_z = zeros({hidden_size, hidden_size + input_size});
    m_w_r = zeros({hidden_size, hidden_size + input_size});
    m_w_h = zeros({hidden_size, hidden_size + input_size});
    m_w_y = zeros({output_size, hidden_size});

    m_b_z = zeros({hidden_size, 1});
    m_b_r = zeros({hidden_size, 1});
    m_b_h = zeros({hidden_size, 1});
    m_b_y = zeros({output_size, 1});

    v_w_z = zeros({hidden_size, hidden_size + input_size});
    v_w_r = zeros({hidden_size, hidden_size + input_size});
    v_w_h = zeros({hidden_size, hidden_size + input_size});
    v_w_y = zeros({output_size, hidden_size});

    v_b_z = zeros({hidden_size, 1});
    v_b_r = zeros({hidden_size, 1});
    v_b_h = zeros({hidden_size, 1});
    v_b_y = zeros({output_size, 1});
}

void gru::train(const tensor& x_train, const tensor& y_train) {
    for (auto i = 1; i <= epochs; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        auto word_embedding = embedding(vocab_size, embedding_dim, x_train);

        auto [x_sequence, concat_sequence, z_t_sequence, r_t_sequence, h_sequence, y_sequence] = forward(word_embedding.dense_vecs, Phase::TRAIN);

        float error = categorical_cross_entropy(transpose(y_train), y_sequence.front());

        tensor d_loss_d_h_t_w_z = zeros({batch_size, hidden_size});
        tensor d_loss_d_h_t_w_r = zeros({batch_size, hidden_size});
        tensor d_loss_d_h_t_w_h = zeros({batch_size, hidden_size});

        tensor d_loss_d_w_z = zeros({hidden_size, hidden_size + input_size});
        tensor d_loss_d_w_r = zeros({hidden_size, hidden_size + input_size});
        tensor d_loss_d_w_h = zeros({hidden_size, hidden_size + input_size});

        tensor d_loss_d_b_z = zeros({hidden_size, 1});
        tensor d_loss_d_b_r = zeros({hidden_size, 1});
        tensor d_loss_d_b_h = zeros({hidden_size, 1});

        float num_samples = static_cast<float>(y_train.shape.front());
        tensor d_loss_d_y = -2.0f / num_samples * (transpose(y_train) - y_sequence.front());

        for (auto j = seq_length; j > 0; --j) {
            if (j == seq_length) {
                tensor d_y_d_h_10 = w_y;

        //         d_loss_d_h_t_w_z = matmul(transpose(d_loss_d_y), d_y_d_h_10);
        //         d_loss_d_h_t_w_r = matmul(transpose(d_loss_d_y), d_y_d_h_10);
        //         d_loss_d_h_t_w_h = matmul(transpose(d_loss_d_y), d_y_d_h_10);
            } else {
        //         d_loss_d_h_t_w_z = matmul(d_loss_d_h_t_w_z * transpose(o_sequence[j] * (1.0f - square(hyperbolic_tangent(c_sequence[j + 1]))) * c_sequence[j + 1] * sigmoid_derivative(z_f_sequence[j])), vslice(w_f, w_f.shape.back() - 1));
        //         d_loss_d_h_t_w_r = matmul(d_loss_d_h_t_w_r * transpose(o_sequence[j] * (1.0f - square(hyperbolic_tangent(c_sequence[j + 1]))) * c_tilde_sequence[j] * sigmoid_derivative(z_i_sequence[j])), vslice(w_i, w_i.shape.back() - 1));
        //         d_loss_d_h_t_w_h = matmul(d_loss_d_h_t_w_h * transpose(o_sequence[j] * (1.0f - square(hyperbolic_tangent(c_sequence[j + 1]))) * i_sequence[j] * (1.0f - square(hyperbolic_tangent(z_c_tilde_sequence[j])))), vslice(w_c, w_c.shape.back() - 1));
            }

        //     d_loss_d_w_z = d_loss_d_w_z + matmul(transpose(d_loss_d_h_t_w_f) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * c_sequence[j] * sigmoid_derivative(z_f_sequence[j - 1]), transpose(concat_sequence[j - 1]));
        //     d_loss_d_w_r = d_loss_d_w_r + matmul(transpose(d_loss_d_h_t_w_i) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * c_tilde_sequence[j - 1] * sigmoid_derivative(z_i_sequence[j - 1]), transpose(concat_sequence[j - 1]));
        //     d_loss_d_w_h = d_loss_d_w_h + matmul(transpose(d_loss_d_h_t_w_c) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * i_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(z_c_tilde_sequence[j - 1]))), transpose(concat_sequence[j - 1]));

        //     d_loss_d_b_z = d_loss_d_b_z + sum(transpose(d_loss_d_h_t_w_f) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * c_sequence[j] * sigmoid_derivative(z_f_sequence[j - 1]), 1);
        //     d_loss_d_b_r = d_loss_d_b_r + sum(transpose(d_loss_d_h_t_w_i) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * c_tilde_sequence[j - 1] * sigmoid_derivative(z_i_sequence[j - 1]), 1);
        //     d_loss_d_b_h = d_loss_d_b_h + sum(transpose(d_loss_d_h_t_w_c) * o_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(c_sequence[j]))) * i_sequence[j - 1] * (1.0f - square(hyperbolic_tangent(z_c_tilde_sequence[j - 1]))), 1);
        }

        tensor d_loss_d_w_y  = matmul(d_loss_d_y, transpose(h_sequence.back()));

        t += 1;

        // m_w_f = beta1 * m_w_f + (1.0f - beta1) * d_loss_d_w_f;
        // m_w_i = beta1 * m_w_i + (1.0f - beta1) * d_loss_d_w_i;
        // m_w_c = beta1 * m_w_c + (1.0f - beta1) * d_loss_d_w_c;
        // m_w_y = beta1 * m_w_y + (1.0f - beta1) * d_loss_d_w_y;

        // m_b_f = beta1 * m_b_f + (1.0f - beta1) * d_loss_d_b_f;
        // m_b_i = beta1 * m_b_i + (1.0f - beta1) * d_loss_d_b_i;
        // m_b_c = beta1 * m_b_c + (1.0f - beta1) * d_loss_d_b_c;
        // m_b_y = beta1 * m_b_y + (1.0f - beta1) * d_loss_d_y;

        // v_w_f = beta2 * v_w_f + (1.0f - beta2) * square(d_loss_d_w_f);
        // v_w_i = beta2 * v_w_i + (1.0f - beta2) * square(d_loss_d_w_i);
        // v_w_c = beta2 * v_w_c + (1.0f - beta2) * square(d_loss_d_w_c);
        // v_w_y = beta2 * v_w_y + (1.0f - beta2) * square(d_loss_d_w_y);

        // v_b_f = beta2 * v_b_f + (1.0f - beta2) * square(d_loss_d_b_f);
        // v_b_i = beta2 * v_b_i + (1.0f - beta2) * square(d_loss_d_b_i);
        // v_b_c = beta2 * v_b_c + (1.0f - beta2) * square(d_loss_d_b_c);
        // v_b_y = beta2 * v_b_y + (1.0f - beta2) * square(d_loss_d_y);

        // tensor m_hat_w_f = m_w_f / (1.0f - powf(beta1, t));
        // tensor m_hat_w_i = m_w_i / (1.0f - powf(beta1, t));
        // tensor m_hat_w_c = m_w_c / (1.0f - powf(beta1, t));
        // tensor m_hat_w_y = m_w_y / (1.0f - powf(beta1, t));

        // tensor m_hat_b_f = m_b_f / (1.0f - powf(beta1, t));
        // tensor m_hat_b_i = m_b_i / (1.0f - powf(beta1, t));
        // tensor m_hat_b_c = m_b_c / (1.0f - powf(beta1, t));
        // tensor m_hat_b_y = m_b_y / (1.0f - powf(beta1, t));

        // tensor v_hat_w_f = v_w_f / (1.0f - powf(beta2, t));
        // tensor v_hat_w_i = v_w_i / (1.0f - powf(beta2, t));
        // tensor v_hat_w_c = v_w_c / (1.0f - powf(beta2, t));
        // tensor v_hat_w_y = v_w_y / (1.0f - powf(beta2, t));

        // tensor v_hat_b_f = v_b_f / (1.0f - powf(beta2, t));
        // tensor v_hat_b_i = v_b_i / (1.0f - powf(beta2, t));
        // tensor v_hat_b_c = v_b_c / (1.0f - powf(beta2, t));
        // tensor v_hat_b_y = v_b_y / (1.0f - powf(beta2, t));

        // w_f = w_f - lr * m_hat_w_f / (sqrt(v_hat_w_f) + epsilon);
        // w_i = w_i - lr * m_hat_w_i / (sqrt(v_hat_w_i) + epsilon);
        // w_c = w_c - lr * m_hat_w_c / (sqrt(v_hat_w_c) + epsilon);
        // w_y = w_y - lr * m_hat_w_y / (sqrt(v_hat_w_y) + epsilon);

        // b_f = b_f - lr * m_hat_b_f / (sqrt(v_hat_b_f) + epsilon);
        // b_i = b_i - lr * m_hat_b_i / (sqrt(v_hat_b_i) + epsilon);
        // b_c = b_c - lr * m_hat_b_c / (sqrt(v_hat_b_c) + epsilon);
        // b_y = b_y - lr * m_hat_b_y / (sqrt(v_hat_b_y) + epsilon);

        // w_z = w_z - lr * d_loss_d_w_z;
        // w_r = w_r - lr * d_loss_d_w_r;
        // w_h = w_h - lr * d_loss_d_w_h;
        w_y = w_y - lr * d_loss_d_w_y;

        // b_z = b_z - lr * d_loss_d_b_z;
        // b_r = b_r - lr * d_loss_d_b_r;
        // b_h = b_h - lr * d_loss_d_b_h;
        b_y = b_y - lr * d_loss_d_y;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remaining_ms = duration - seconds;

        std::cout << "Epoch " << i << "/" << epochs << std::endl << seconds.count() << "s " << remaining_ms.count() << "ms/step - loss: " << 0.0f << std::endl;
    }
}

float gru::evaluate(const tensor& x, const tensor& y) {
    auto [x_sequence, concat_sequence, z_t_sequence, r_t_sequence, h_sequence, y_sequence] = forward(x, Phase::TEST);
    // return loss(transpose(y), y_sequence.front());
    return 0.0f;
}

tensor gru::predict(const tensor& x) {
    auto [x_sequence, concat_sequence, z_t_sequence, r_t_sequence, h_sequence, y_sequence] = forward(x, Phase::TEST);
    return transpose(y_sequence.front());
}

std::array<std::vector<tensor>, 6> gru::forward(const tensor& x, enum Phase phase) {
    std::vector<tensor> x_sequence;
    std::vector<tensor> concat_sequence;
    std::vector<tensor> z_t_sequence;
    std::vector<tensor> r_t_sequence;
    std::vector<tensor> h_sequence;
    std::vector<tensor> y_sequence;

    if (phase == Phase::TRAIN)
        batch_size = 60841;
    else
        batch_size = 15211;

    tensor ones = tensor({hidden_size, batch_size}, {1.0f});

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

        tensor z_t_z = matmul(w_z, concat_t) + b_z;
        tensor z_t = sigmoid(z_t_z);

        tensor r_t_z = matmul(w_r, concat_t) + b_r;
        tensor r_t = sigmoid(r_t_z);

        tensor h_hat_t_z = matmul(w_h, vstack({r_t * h_t, transpose(x_t)})) + b_h;
        tensor h_hat_t = hyperbolic_tangent(h_hat_t_z);

        h_t = (ones - z_t) * h_t + z_t * h_hat_t;

        tensor y_t_z = matmul(w_y, h_t) + b_y;
        tensor y_t = softmax(y_t_z);

        // std::cout << h_t.get_shape() << std::endl;
        // std::cout << concat_t.get_shape() << std::endl;
        // std::cout << r_t.get_shape() << std::endl;
        // std::cout << x_t.get_shape() << std::endl;

        // =====================================================================================================================================
        // dL/dw_h: (dL/dy * dy/dh_10) * dh_10/do_10 * do_t10/dw_o
        // dL/dw_r: (dL/dy * dy/dh_10) * dh_10/do_10 * do_t10/dw_o
        // dL/dw_z: (dL/dy * dy/dh_10) * dh_10/do_10 * do_t10/dw_o
        // dL/dembedding:
        // =====================================================================================================================================

        // =====================================================================================================================================
        // Check forward pass on https://en.wikipedia.org/wiki/Long_short-term_memory to remind myself that way to compute gradients make sense.
        // (dL/dy * dy/dh_10) * dh_10/do_10 * do_t10/dw_o
        // (dL/dy * dy/dh_10 * dh_10/do_10 * do_10/dh_9) * dh_9/do_9 * do_9/dw_o
        // (dL/dy * dy/dh_10 * dh_10/do_10 * do_10/dh_9 * dh_9/do_9 * do_9/dh_8) * dh_8/do_8 * do_8/dw_o
        // =====================================================================================================================================
        // (dL/dy * dy/dh_10) * dh_10/dc_10 * dc_10/dc_tilde_10 * dc_tilde_10/dw_c
        // (dL/dy * dy/dh_10 * dh_10/dc_10 * dc_10/dc_tilde_10 * dc_tilde_10/dh_9) * dh_9/dc_9 * dc_9/dc_tilde_9 * dc_tilde9/dw_c

        // dh10/do10 * do10/wo
        // dh10/do10 * do10/dh9 * dh9/do9 * do9/wo

        // dh10/dc10 * dc10/d~c10 * d~c10/w_c
        // dh10/dc10 * dc10/d~c10 * d~c10/dh9 * dh9/dc9 * dc9/d~c9 * d~c9/wc
        // =====================================================================================================================================
        // (dL/dy * dy/dh_10) * dh_10/dc_10 * dc_10/di_10 * di_10/dw_i
        // (dL/dy * dy/dh_10 * dh_10/dc_10 * dc_10/di_10 * di_10/dh_9) * dh_9/dc_9 * dc_9/di_9 * di_9/dw_i
        // =====================================================================================================================================
        // (dL/dy * dy/dh_10) * dh_10/dc_10 * dc_10/df_10 * df_10/dw_f
        // (dL/dy * dy/dh_10 * dh_10/dc_10 * dc_10/df_10 * df_10/dh_9) * dh_9/dc_9 * dc_9/df_9 * df_9/dw_f
        // =====================================================================================================================================

        x_sequence.push_back(x_t);
        concat_sequence.push_back(concat_t);
        z_t_sequence.push_back(z_t);
        r_t_sequence.push_back(r_t);
        h_sequence.push_back(h_t);

        if (i == seq_length - 1)
            y_sequence.push_back(y_t);
    }

    std::array<std::vector<tensor>, 6> sequences;

    sequences[0] = x_sequence;
    sequences[1] = concat_sequence;
    sequences[2] = z_t_sequence;
    sequences[3] = r_t_sequence;
    sequences[4] = h_sequence;
    sequences[5] = y_sequence;

    return sequences;
}

embedding::embedding(const size_t vocab_size, const size_t embedding_dim, const tensor& t) {
    for (auto i = 0; i < t.size; ++i)
        assert(t[i] < vocab_size);

    mat = uniform_dist({vocab_size, embedding_dim});

    dense_vecs = zeros({t.shape.front(), t.shape.back(), embedding_dim});

    for (auto i = 0; i < t.size; ++i) {
        auto a = slice(mat, t[i], 1);

        for (auto j = 0; j < a.size; ++j)
            dense_vecs[embedding_dim * i + j] = a[j];
    }
}

tensor text_vectorization(const std::vector<std::string>& vocab, const std::vector<std::string>& in, size_t max_tokens, const size_t max_len) {
    assert(max_tokens > 2);

    std::unordered_map<std::string, float> vocab_map;

    for (auto text : vocab) {
        auto tokens = tokenizer(text);

        for (auto token : tokens) {
            token = lower(token);
            token = regex_replace(token, "[\".,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", "");

            if (vocab_map.find(token) != vocab_map.end())
                vocab_map[token] += 1.0f;
            else
                vocab_map.insert(std::pair<std::string, float>(token, 1.0f));
        }
    }

    std::vector<std::pair<std::string, float>> vocab_vec(vocab_map.begin(), vocab_map.end());

    std::sort(vocab_vec.begin(), vocab_vec.end(), [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b) {
        if (a.second != b.second)
            return a.second > b.second;
        else
            return a.first > b.first;
    });

    vocab_vec.insert(vocab_vec.begin(), std::pair<std::string, float>("[UNK]", 1.0f));
    vocab_vec.insert(vocab_vec.begin(), std::pair<std::string, float>("", 0.0f));

    // for (auto i = 0; i < 20; ++i)
    //   std::cout << vocab_vec[i].first << " " << vocab_vec[i].second << std::endl;

    tensor t_new = zeros({in.size(), max_len});

    size_t idx = 0;
    const float oov_token = vocab_vec[1].second;

    if (max_tokens > vocab_vec.size())
        max_tokens = vocab_vec.size();

    for (auto i = 0; i < in.size(); ++i) {
        auto words = tokenizer(in[i]);

        if (i != 0)
            idx = i * max_len;

        size_t words_processed = 0;

        for (auto word : words) {
            ++words_processed;

            word = lower(word);
            word = regex_replace(word, "[\".,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", "");

            bool found = false;

            for (auto k = 0; k < max_tokens; ++k) {
                if (word == vocab_vec[k].first) {
                    t_new[idx] = k;
                    found = true;
                    break;
                }
            }

            if (!found)
                t_new[idx] = oov_token;

            if (words_processed == max_len)
              break;

            ++idx;
        }
    }

    return t_new;
}