#pragma once

#include "math.hpp"
#include "tensor.h"

class embedding {
  public:
    tensor mat;
    tensor dense_vecs;

    embedding(const size_t vocab_size, const size_t embedding_dim, const tensor& t);
};

class min_max_scaler {
  private:
    tensor data_min;
    tensor data_max;
    bool is_fitted;

  public:
    min_max_scaler() : data_min(), data_max(), is_fitted(false) {}

    void fit(const tensor& data) {
        data_min = min(data);
        data_max = max(data, 0);
        is_fitted = true;
    }

    tensor transform(const tensor& data) {
        if (!is_fitted)
            std::cerr << "Scaler not fitted yet." << std::endl;

        return (data - data_min) / (data_max - data_min);
    }

    tensor inverse_transform(const tensor& scaled_data) {
       if (!is_fitted)
            std::cerr << "Scaler not fitted yet." << std::endl;

        return scaled_data * (data_max - data_min) + data_min;
    }

};

tensor text_vectorization(const std::vector<std::string>& vocab, const std::vector<std::string>& in, size_t max_tokens, const size_t max_len);