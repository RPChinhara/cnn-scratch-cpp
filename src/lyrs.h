#pragma once

#include "arrs.h"
#include "preproc.h"
#include "tensor.h"

#include <cassert>
#include <functional>
#include <random>
#include <tuple>
#include <unordered_map>
#include <vector>

class tensor;

using act_func = std::function<tensor(const tensor&)>;
using loss_func = std::function<float(const tensor&, const tensor&)>;
using metric_func = std::function<float(const tensor&, const tensor&)>;

class cnn2d {
  private:
    std::vector<size_t> filters;
    float lr;

    std::vector<tensor> forward(const tensor &input, const std::vector<tensor> &kernel, const size_t stride);

  public:
    cnn2d(const std::vector<size_t> &filters, float const lr);
    void train(const tensor &x_train, const tensor &y_train, const tensor &x_val, const tensor &y_val);
    void predict(const tensor &xTest, const tensor &yTest);
};

class cnn_lstm_2d {
  private:
    std::vector<size_t> filters;
    float lr;

    std::vector<tensor> forward(const tensor &input, const std::vector<tensor> &kernel, const size_t stride);

  public:
    cnn_lstm_2d(const std::vector<size_t> &filters, float const lr);
    void train(const tensor &x_train, const tensor &y_train, const tensor &x_val, const tensor &y_val);
    void predict(const tensor &xTest, const tensor &yTest);
};

class gru {
  private:
    size_t num_ins = 10;
    size_t num_hiddens = 20;
    size_t batch_size = 20;

    tensor u_z;
    tensor u_r;
    tensor u_h;

    tensor w_z;
    tensor w_r;
    tensor w_h;

    tensor b_z;
    tensor b_r;
    tensor b_h;

    tensor h;

    std::pair<std::vector<tensor>, std::vector<tensor>> init_params();
    std::vector<tensor> forward(const tensor &x);

  public:
    gru(const size_t units);
};

class lstm {
  private:
    float lr;
    loss_func loss;
    size_t batch_size = 8316;
    size_t epochs = 10;

    size_t in_size = 1;
    size_t hidden_size = 50;
    size_t out_size = 1;
    size_t seq_length = 10;

    tensor w_xh;
    tensor w_hh;
    tensor w_hy;
    tensor b_h;
    tensor b_y;

    std::vector<tensor> forward(const tensor &x);

  public:
    lstm(const size_t lr, loss_func loss);
    void train(const tensor &x_train, const tensor &y_train, const tensor &x_val, const tensor &y_val);
};

class nn {
  private:
    std::vector<size_t> lyrs;
    std::vector<act_func> activations;
    loss_func loss;
    metric_func metric;
    float lr;
    size_t epochs = 200;
    size_t batch_size = 10;
    float momentum = 0.1f;

    std::pair<std::vector<tensor>, std::vector<tensor>> w_b;
    std::pair<std::vector<tensor>, std::vector<tensor>> w_b_momentum;

    std::pair<std::vector<tensor>, std::vector<tensor>> init_params();
    std::vector<tensor> forward(const tensor &x, const std::vector<tensor> &w, const std::vector<tensor> &b);

  public:
    nn(const std::vector<size_t> &lyrs, const std::vector<act_func> &activations, const loss_func &loss, const metric_func &metric, const float lr);
    void train(const tensor &x_train, const tensor &y_train, const tensor &x_val, const tensor &y_val);
    float evaluate(const tensor &x, const tensor &y);
    tensor predict(const tensor &x);
};

class rnn {
  private:
    act_func activation;
    loss_func loss;
    float lr;
    size_t epochs = 150;
    size_t batch_size = 8317;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-7f;
    size_t t = 0;

    size_t input_size = 1;
    size_t hidden_size = 50;
    size_t output_size = 1;
    size_t seq_length = 10;

    tensor h_t;
    tensor w_xh;
    tensor w_hh;
    tensor w_hy;
    tensor b_h;
    tensor b_y;

    tensor m_w_xh;
    tensor m_w_hh;
    tensor m_w_hy;
    tensor m_b_h;
    tensor m_b_y;

    tensor v_w_xh;
    tensor v_w_hh;
    tensor v_w_hy;
    tensor v_b_h;
    tensor v_b_y;

    enum Phase {
      TRAIN,
      TEST
    };

    std::tuple<std::vector<tensor>, std::vector<tensor>, std::vector<tensor>, std::vector<tensor>> forward(const tensor &x, enum Phase phase);

  public:
    rnn(const act_func &activation, const loss_func &loss, const float lr);
    void train(const tensor &x_train, const tensor &y_train);
    float evaluate(const tensor &x, const tensor &y);
    tensor predict(const tensor &x);
};

tensor embedding(const size_t vocab_size, const size_t cols, const tensor &ind);

template <typename T>
tensor text_vectorization(const std::vector<T> &vocab, const std::vector<T> &in, size_t max_tokens, const size_t max_len) {
    assert(max_tokens > 2);

    std::unordered_map<T, float> vocab_map;

    for (auto text : vocab) {
        auto tokens = tokenizer(text);

        for (auto token : tokens) {
            if (vocab_map.find(token) != vocab_map.end())
                vocab_map[token] += 1.0f;
            else
                vocab_map.insert(std::pair<T, float>(token, 1.0f));
        }
    }

    std::vector<std::pair<T, float>> vocab_vec(vocab_map.begin(), vocab_map.end());

    std::sort(vocab_vec.begin(), vocab_vec.end(), [](const std::pair<T, float> &a, const std::pair<T, float> &b) {
        if (a.second != b.second)
            return a.second > b.second;
        else
            return a.first > b.first;
    });

    vocab_vec.insert(vocab_vec.begin(), std::pair<T, float>("[UNK]", 1.0f));
    vocab_vec.insert(vocab_vec.begin(), std::pair<T, float>("", 0.0f));

    tensor t_new = zeros({in.size(), max_len});

    size_t idx = 0;
    const float oov_token = vocab_vec[1].second;

    if (max_tokens > vocab_vec.size())
        max_tokens = vocab_vec.size();

    for (auto i = 0; i < in.size(); ++i) {
        auto words = tokenizer(in[i]);

        if (i != 0)
            idx = i * max_len;

        for (auto word : words) {
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

            ++idx;
        }
    }

    return t_new;
}