#pragma once

#include "math.h"
#include "tensor.h"

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

tensor text_vectorization(const std::vector<std::string>& vocab, const std::vector<std::string>& in, size_t max_tokens, const size_t max_len);
tensor layer_normalization(const tensor& x);