#pragma once

#include "act.h"

#include <string>
#include <vector>

class ten;

class cnn2d
{
  public:
    cnn2d(const std::vector<size_t> &filters, float const lr);
    void train(const ten &xTrain, const ten &yTrain, const ten &xVal, const ten &yVal);
    void pred(const ten &xTest, const ten &yTest);

  private:
    std::vector<ten> forward(const ten &input, const std::vector<ten> &kernel, const size_t stride);

    std::vector<size_t> filters;
    float lr;
};

class gru
{
  public:
    gru(const size_t units);

  private:
    std::pair<std::vector<ten>, std::vector<ten>> init_params();
    std::vector<ten> forward(const ten &x, const ten &h_prev);
};

class nn
{
  public:
    nn(const std::vector<size_t> &lyrs, const std::vector<act_enum> &act_types, float const lr);
    void train(const ten &x_train, const ten &y_train, const ten &x_val, const ten &y_val);
    void pred(const ten &x_test, const ten &y_test);

  private:
    std::pair<std::vector<ten>, std::vector<ten>> init_params();
    std::vector<ten> forward(const ten &x, const std::vector<ten> &w, const std::vector<ten> &b);

    std::vector<size_t> lyrs;
    std::vector<act_enum> act_types;
    std::pair<std::vector<ten>, std::vector<ten>> w_b;
    std::pair<std::vector<ten>, std::vector<ten>> w_b_mom;
    std::vector<ten> a;
    size_t batch_size = 10;
    size_t epochs = 200;
    float lr;
    float grad_clip_threshold = 8.0f;
    float mom = 0.1f;
    size_t patience = 4;
};

ten embedding(const size_t vocab_size, const size_t cols, const ten &ind);
ten text_vectorization(const std::vector<std::wstring> &vocab, const std::vector<std::wstring> &in);