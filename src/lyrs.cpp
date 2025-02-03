#include "lyrs.h"
#include "acts.h"
#include "arrs.h"
#include "linalg.h"
#include "rand.h"
#include "strings.h"

#include <random>
#include <unordered_map>

embedding::embedding(const size_t vocab_size, const size_t embedding_dim) {
    this->embedding_dim = embedding_dim;
    embedding_mat = uniform_dist({vocab_size, embedding_dim});
}

tensor embedding::adapt(const tensor& t) {
    tensor embedded_tokens = zeros({t.shape.front(), t.shape.back(), embedding_dim});

    for (size_t i = 0; i < t.size; ++i) {
        tensor embedding_vec = slice(embedding_mat, t[i], 1);

        for (size_t j = 0; j < embedding_vec.size; ++j) {
            embedded_tokens[i * embedding_dim + j] = embedding_vec[j];
        }
    }

    return embedded_tokens;
}

positional_encoding::positional_encoding(const size_t seq_len, const size_t dim) {
    pe = zeros({seq_len, dim});

    for (size_t i = 0; i < seq_len; ++i) {
        for (size_t j = 0; j < dim / 2; ++j) {
            float denominator = pow(10000, 2.0f * j / dim);
            pe(i, 2 * j) = sin(i / denominator);
            pe(i, 2 * j + 1) = cos(i / denominator);
        }
    }
}

tensor positional_encoding::adapt(tensor& embedded_tokens) {
    size_t idx = 0;
    const size_t block_size = embedded_tokens.shape[1] * embedded_tokens.shape[2];

    for (size_t k = 0; k < embedded_tokens.size; ++k) {
        embedded_tokens[k] += pe[idx];
        idx = (k + 1) % block_size == 0 ? 0 : idx + 1;
    }

    return embedded_tokens;
}

tensor text_vectorization(const std::vector<std::string>& vocab, const std::vector<std::string>& in, size_t max_tokens, const size_t max_len) {
    std::unordered_map<std::string, float> vocab_map;

    for (auto text : vocab) {
        auto tokens = tokenizer(text);

        for (auto token : tokens) {
            token = lower(token);
            token = regex_replace(token, "[\".,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", "");

            if (vocab_map.find(token) != vocab_map.end())
                vocab_map[token] += 1.0f;
            else
                vocab_map.insert(std::pair<std::string, float>(token, 1.0f));
        }
    }

    std::vector<std::pair<std::string, float>> vocab_vec(vocab_map.begin(), vocab_map.end());

    std::sort(vocab_vec.begin(), vocab_vec.end(), [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b) {
        if (a.second != b.second)
            return a.second > b.second;
        else
            return a.first > b.first;
    });

    vocab_vec.insert(vocab_vec.begin(), std::pair<std::string, float>("[UNK]", 1.0f));
    vocab_vec.insert(vocab_vec.begin(), std::pair<std::string, float>("", 0.0f));

    // for (size_t i = 0; i < vocab_vec.size(); ++i)
    //   std::cout << vocab_vec[i].first << " " << vocab_vec[i].second << std::endl;

    tensor t_new = zeros({in.size(), max_len});

    size_t idx = 0;
    const float oov_token = vocab_vec[1].second;

    // NOTE: In TensorFlow, the max_tokens (or vocabulary size) is max_tokens - 2 when output_mode == "int", because 0 is reserved for padding tokens and 1 is reserved for OOV (out-of-vocabulary) tokens.
    if (max_tokens > vocab_vec.size())
        max_tokens = vocab_vec.size();

    for (auto i = 0; i < in.size(); ++i) {
        auto words = tokenizer(in[i]);

        if (i != 0)
            idx = i * max_len;

        size_t words_processed = 0;

        for (auto word : words) {
            ++words_processed;

            word = lower(word);
            word = regex_replace(word, "[\".,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", "");

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

            if (words_processed == max_len)
              break;

            ++idx;
        }
    }

    return t_new;
}

tensor layer_normalization(const tensor& x) {
    const size_t features = x.shape.back();

    float epsilon = 1e-5f;
    tensor gamma = fill({1, features}, 1.0f);
    tensor beta = zeros({1, features});

    tensor average = mean(x);
    tensor var = variance(x);

    tensor x_hat = (x - average) / sqrt(var + epsilon);

    tensor y = gamma * x_hat + beta;

    return y;
}

tensor multihead_attention(const tensor& x, std::vector<std::vector<tensor>> w, const size_t seq_len, const size_t d_model, const size_t num_heads) {
    size_t batch_size = x.shape.front();
    size_t head_dim = (num_heads == 1) ? d_model : d_model / num_heads;

    tensor outputs = zeros({batch_size, seq_len, d_model});

    // TODO: I want to make a operator extract a matrix from 3D or 4D tensor -> this is fundamentally same as slicing 3D/4D tensor to extract matrices so...
    // TODO: Should I modify matmul() to support 3D or even 4D tensors like NumPy does? There's no concept of 3D matrix multiplication in traditional math, so it would essentially be the same whether the 3D handling is done in matmul() or at this level. However, for now, handle it as I always do when dealing with 3D/4D tensors.

    // x: (10, 25, 128) or (8, 25, 128)
    // x_mat: (25, 128)

    for (size_t i = 0; i < batch_size; ++i) {
        tensor x_mat = slice(x, i * seq_len, seq_len);
        std::vector<tensor> attention_heads;

        // TODO: I think I need Multithreading for this?
        for (size_t j = 0; j < num_heads; ++j) {
            tensor q_mat = matmul(x_mat, w[j][0]);
            tensor k_mat = matmul(x_mat, w[j][1]);
            tensor v_mat = matmul(x_mat, w[j][2]);

            tensor attention_scores = matmul(q_mat, transpose(k_mat));
            tensor scaled_scores = attention_scores / sqrt(head_dim);
            tensor attention_weights = softmax(scaled_scores);

            // TODO: Add mask here to attention_weights?

            tensor weighted_sum = matmul(attention_weights, v_mat);
            attention_heads.push_back(weighted_sum);
        }

        tensor concatenated_heads = concat(attention_heads, 1);
        tensor output = matmul(concatenated_heads, w[w.size() - 1][0]);

        for (size_t j = 0; j < output.size; ++j)
            outputs[i * output.size + j] = output[j];
    }

    return outputs;
}