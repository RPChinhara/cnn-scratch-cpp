#pragma once

#include "tensor.h"

class embedding {
  public:
    tensor mat;
    tensor dense_vecs;

    embedding(const size_t vocab_size, const size_t embedding_dim, const tensor& t);
};

tensor text_vectorization(const std::vector<std::string>& vocab, const std::vector<std::string>& in, size_t max_tokens, const size_t max_len);