#pragma once

#include "math.h"
#include "tensor.h"

class embedding {
  public:
    tensor embedding_mat;
    tensor embedded_tokens;

    embedding(const size_t vocab_size, const size_t embedding_dim, const tensor& t);
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