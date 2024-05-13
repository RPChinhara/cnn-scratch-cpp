#pragma once

#include "act.h"
#include "ten.h"

#include <string>
#include <vector>

class ten;

class cnn2d
{
  private:
    std::vector<size_t> filters;
    float lr;

    std::vector<ten> forward(const ten &input, const std::vector<ten> &kernel, const size_t stride);

  public:
    cnn2d(const std::vector<size_t> &filters, float const lr);
    void pred(const ten &xTest, const ten &yTest);
    void train(const ten &xTrain, const ten &yTrain, const ten &xVal, const ten &yVal);
};

class fnn
{
  private:
    std::vector<size_t> lyrs;
    size_t batch_size = 10;
    float lr;
    size_t epochs = 200;
    float grad_clip_threshold = 8.0f;
    float mom = 0.1f;
    size_t patience = 4;

    std::vector<ten> a;
    std::vector<act_enum> act_types;
    std::pair<std::vector<ten>, std::vector<ten>> w_b;
    std::pair<std::vector<ten>, std::vector<ten>> w_b_mom;

    std::vector<ten> forward(const ten &x, const std::vector<ten> &w, const std::vector<ten> &b);
    std::pair<std::vector<ten>, std::vector<ten>> init_params();

  public:
    fnn(const std::vector<size_t> &lyrs, const std::vector<act_enum> &act_types, float const lr);
    void pred(const ten &x_test, const ten &y_test);
    void train(const ten &x_train, const ten &y_train, const ten &x_val, const ten &y_val);
};

class gru
{
  private:
    size_t num_ins = 10;
    size_t num_hiddens = 20;
    size_t batch_size = 20;

    ten u_z;
    ten u_r;
    ten u_h;

    ten w_z;
    ten w_r;
    ten w_h;

    ten b_z;
    ten b_r;
    ten b_h;

    ten h;

    std::vector<ten> forward(const ten &x);
    std::pair<std::vector<ten>, std::vector<ten>> init_params();

  public:
    gru(const size_t units);
};

class rnn
{
  private:
    size_t hidden_size;
    size_t vocab_size;
    size_t seq_length;
    float lr;

    ten U;
    ten V;
    ten W;
    ten b;
    ten c;

    std::vector<ten> forward(const ten &x);

  public:
    rnn(const size_t hidden_size, const size_t vocab_size, const size_t seq_length, const size_t lr);
};

ten embedding(const size_t vocab_size, const size_t cols, const ten &ind);
ten text_vectorization(const std::vector<std::wstring> &vocab, const std::vector<std::wstring> &in);