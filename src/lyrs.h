#pragma once

#include "arrs.h"
#include "preproc.h"
#include "tensor.h"

#include <array>
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

class gru {
  private:
    float lr;
    size_t batch_size;
    size_t epochs = 250;

    size_t vocab_size;
    size_t embedding_dim = 50;

    size_t seq_length = 25;
    size_t input_size = embedding_dim;
    size_t hidden_size = 50;
    size_t output_size = 25;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-7f;
    size_t t = 0;

    tensor w_z;
    tensor w_r;
    tensor w_h;
    tensor w_y;

    tensor b_z;
    tensor b_r;
    tensor b_h;
    tensor b_y;

    tensor m_w_z;
    tensor m_w_r;
    tensor m_w_h;
    tensor m_w_y;

    tensor m_b_z;
    tensor m_b_r;
    tensor m_b_h;
    tensor m_b_y;

    tensor v_w_z;
    tensor v_w_r;
    tensor v_w_h;
    tensor v_w_y;

    tensor v_b_z;
    tensor v_b_r;
    tensor v_b_h;
    tensor v_b_y;

    enum Phase {
      TRAIN,
      TEST
    };

    std::array<std::vector<tensor>, 6> forward(const tensor& x, enum Phase phase);

  public:
    gru(const float lr, const size_t vocab_size);
    void train(const tensor& x_train, const tensor& y_train);
    float evaluate(const tensor& x, const tensor& y);
    tensor predict(const tensor& x);
};

class lstm {
  private:
    loss_func loss;
    float lr;
    size_t batch_size;
    size_t epochs = 250;

    size_t seq_length = 10;
    size_t input_size = 1;
    size_t hidden_size = 50;
    size_t output_size = 1;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-7f;
    size_t t = 0;

    tensor w_f;
    tensor w_i;
    tensor w_c;
    tensor w_o;
    tensor w_y;

    tensor b_f;
    tensor b_i;
    tensor b_c;
    tensor b_o;
    tensor b_y;

    tensor m_w_f;
    tensor m_w_i;
    tensor m_w_c;
    tensor m_w_o;
    tensor m_w_y;

    tensor m_b_f;
    tensor m_b_i;
    tensor m_b_c;
    tensor m_b_o;
    tensor m_b_y;

    tensor v_w_f;
    tensor v_w_i;
    tensor v_w_c;
    tensor v_w_o;
    tensor v_w_y;

    tensor v_b_f;
    tensor v_b_i;
    tensor v_b_c;
    tensor v_b_o;
    tensor v_b_y;

    enum Phase {
      TRAIN,
      TEST
    };

    std::array<std::vector<tensor>, 12> forward(const tensor& x, enum Phase phase);

  public:
    lstm(const loss_func &loss, const float lr);
    void train(const tensor& x_train, const tensor& y_train);
    float evaluate(const tensor& x, const tensor& y);
    tensor predict(const tensor& x);
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
    std::pair<std::vector<tensor>, std::vector<tensor>> forward(const tensor& x, const std::vector<tensor> &w, const std::vector<tensor> &b);

  public:
    nn(const std::vector<size_t> &lyrs, const std::vector<act_func> &activations, const loss_func &loss, const metric_func &metric, const float lr);
    void train(const tensor& x_train, const tensor& y_train, const tensor& x_val, const tensor& y_val);
    float evaluate(const tensor& x, const tensor& y);
    tensor predict(const tensor& x);
};

class rnn {
  private:
    act_func activation;
    loss_func loss;
    float lr;
    size_t epochs = 150;
    size_t batch_size = 8317;

    size_t seq_length = 10;
    size_t input_size = 1;
    size_t hidden_size = 50;
    size_t output_size = 1;

    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float epsilon = 1e-7f;
    size_t t = 0;

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

    std::tuple<std::vector<tensor>, std::vector<tensor>, std::vector<tensor>, std::vector<tensor>> forward(const tensor& x, enum Phase phase);

  public:
    rnn(const act_func &activation, const loss_func &loss, const float lr);
    void train(const tensor& x_train, const tensor& y_train);
    float evaluate(const tensor& x, const tensor& y);
    tensor predict(const tensor& x);
};

class embedding {
  public:
    tensor mat;
    tensor dense_vecs;

    embedding(const size_t vocab_size, const size_t embedding_dim, const tensor& t);
};

tensor text_vectorization(const std::vector<std::string> &vocab, const std::vector<std::string> &in, size_t max_tokens, const size_t max_len);