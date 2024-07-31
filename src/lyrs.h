#pragma once

#include "act.h"
#include "arrs.h"
#include "preproc.h"
#include "ten.h"

#include <cassert>
#include <functional>
#include <random>
#include <unordered_map>
#include <vector>

class ten;

class cnn2d {
  private:
    std::vector<size_t> filters;
    float lr;

    std::vector<ten> forward(const ten &input, const std::vector<ten> &kernel, const size_t stride);

  public:
    cnn2d(const std::vector<size_t> &filters, float const lr);
    void train(const ten &x_train, const ten &y_train, const ten &x_val, const ten &y_val);
    void predict(const ten &xTest, const ten &yTest);
};

class gru {
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

    std::pair<std::vector<ten>, std::vector<ten>> init_params();
    std::vector<ten> forward(const ten &x);

  public:
    gru(const size_t units);
};

class nn {
  private:
    std::vector<ten> a;
    std::vector<act_type> act_types;
    size_t batch_size = 10;
    size_t epochs = 200;
    float grad_clip_threshold = 8.0f;
    float lr;
    std::function<float(const ten&, const ten&)> loss;
    std::function<float(const ten&, const ten&)> metric;
    std::vector<size_t> lyrs;
    float mom = 0.1f;

    std::pair<std::vector<ten>, std::vector<ten>> w_b;
    std::pair<std::vector<ten>, std::vector<ten>> w_b_mom;

    std::pair<std::vector<ten>, std::vector<ten>> init_params();
    std::vector<ten> forward(const ten &x, const std::vector<ten> &w, const std::vector<ten> &b);

  public:
    nn(const std::vector<size_t> &lyrs, const std::vector<act_type> &act_types, float const lr, std::function<float(const ten&, const ten&)> loss, std::function<float(const ten&, const ten&)> metric);
    void train(const ten &x_train, const ten &y_train, const ten &x_val, const ten &y_val);
    float evaluate(const ten &x, const ten &y);
    ten predict(const ten &x);
};

class rnn {
  private:
    float lr;
    std::function<float(const ten&, const ten&)> loss;
    size_t batch_size = 8316;
    size_t epochs = 10;

    size_t in_size = 1;
    size_t hidden_size = 50;
    size_t out_size = 1;
    size_t seq_length = 10;

    ten w_ih;
    ten w_hh;
    ten w_ho;
    ten b_h;
    ten b_o;

    std::vector<ten> forward(const ten &x);

  public:
    rnn(const size_t lr, std::function<float(const ten&, const ten&)> loss);
    void train(const ten &x_train, const ten &y_train, const ten &x_val, const ten &y_val);
};

ten embedding(const size_t vocab_size, const size_t cols, const ten &ind);

template <typename T>
ten text_vectorization(const std::vector<T> &vocab, const std::vector<T> &in, size_t max_tokens, const size_t max_len) {
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

    // size_t num_vocab = 200;
    // for (auto i = 0; i < num_vocab; ++i)
    //     std::cout << i << ": " << vocab_vec[i].first << " " << vocab_vec[i].second << std::endl;

    ten t_new = zeros({in.size(), max_len});

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