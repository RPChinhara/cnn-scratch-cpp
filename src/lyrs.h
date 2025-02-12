#pragma once

#include "math.h"
#include "tensor.h"

class min_max_scaler {
  private:
    tensor data_min;
    tensor data_max;

  public:
    min_max_scaler() = default;

    void fit(const tensor& data);
    tensor transform(const tensor& data);
    tensor inverse_transform(const tensor& scaled_data);
};

class text_vectorizer {
  public:
    text_vectorizer(size_t vocab_size, size_t seq_len) : vocab_size(vocab_size), seq_len(seq_len) {}
    void build_vocab(const std::vector<std::string>& data);
    tensor vectorize(const std::vector<std::string>& input);

  private:
    size_t vocab_size;
    size_t seq_len;
    std::vector<std::pair<std::string, float>> vocab_vec;
};

class embedding {
  public:
    size_t embedding_dim;
    tensor embedding_mat;

    embedding(const size_t vocab_size, const size_t embedding_dim);
    tensor adapt(const tensor& t);
};

class positional_encoding {
  private:
    // NOTE: They say that these days, pe is also trained similar to training embedding matrix, but not sure.
    tensor pe;

  public:
    positional_encoding(const size_t seq_len, const size_t dim);
    tensor adapt(tensor& embedded_tokens);
};

tensor layer_normalization(const tensor& x);
tensor multihead_attention(const tensor& x, const std::vector<std::vector<tensor>>& w, size_t seq_len, size_t d_model, size_t num_heads, bool mask = false);