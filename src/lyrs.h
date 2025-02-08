#pragma once

#include "math.h"
#include "tensor.h"

class min_max_scaler {
  private:
    tensor data_min;
    tensor data_max;

  public:
    min_max_scaler() = default;

    void fit(const tensor& data) {
        data_min = min(data);
        data_max = max(data, 0);
    }

    tensor transform(const tensor& data) {
        return (data - data_min) / (data_max - data_min);
    }

    tensor inverse_transform(const tensor& scaled_data) {
        return scaled_data * (data_max - data_min) + data_min;
    }

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

class text_vectorization {
  public:
    text_vectorization(size_t vocab_size, size_t seq_len) : vocab_size(vocab_size), seq_len(seq_len) {}
    void build_vocab(const std::vector<std::string>& data);
    tensor vectorize(const std::vector<std::string>& input);

  private:
    size_t vocab_size;
    size_t seq_len;
    std::vector<std::pair<std::string, float>> vocab_vec;
};

tensor layer_normalization(const tensor& x);
tensor multihead_attention(const tensor& x, std::vector<std::vector<tensor>> w, const size_t seq_len, const size_t d_model, const size_t num_heads);