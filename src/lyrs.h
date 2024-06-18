#pragma once

#include "act.h"
#include "arrs.h"
#include "preproc.h"
#include "ten.h"

#include <random>
#include <unordered_map>
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

class nn
{
  private:
    std::vector<ten> a;
    std::vector<act_type> act_types;
    size_t batch_size = 10;
    size_t epochs = 200;
    float grad_clip_threshold = 8.0f;
    float lr;
    std::vector<size_t> lyrs;
    float mom = 0.1f;
    size_t patience = 4;

    std::pair<std::vector<ten>, std::vector<ten>> w_b;
    std::pair<std::vector<ten>, std::vector<ten>> w_b_mom;

    std::vector<ten> forward(const ten &x, const std::vector<ten> &w, const std::vector<ten> &b);
    std::pair<std::vector<ten>, std::vector<ten>> init_params();

  public:
    nn(const std::vector<size_t> &lyrs, const std::vector<act_type> &act_types, float const lr);
    void pred(const ten &x_test, const ten &y_test);
    void train(const ten &x_train, const ten &y_train, const ten &x_val, const ten &y_val);
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

template <typename T>
ten text_vectorization(const std::vector<T> &vocab, const std::vector<T> &in, size_t max_tokens, const size_t max_len)
{
    std::unordered_map<T, float> vocab_map;

    for (auto text : vocab)
    {
        auto tokens = tokenizer(text);

        for (auto token : tokens)
        {
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

    ten t_new = zeros({in.size(), max_len});

    size_t idx = 0;
    const float oov_token = vocab_vec[1].second;

    if (max_tokens > vocab_vec.size())
        max_tokens = vocab_vec.size();

    for (auto i = 0; i < in.size(); ++i)
    {
        auto words = tokenizer(in[i]);

        if (i != 0)
            idx = i * max_len;

        for (auto word : words)
        {
            bool found = false;

            for (auto k = 0; k < max_tokens; ++k)
            {
                if (word == vocab_vec[k].first)
                {
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