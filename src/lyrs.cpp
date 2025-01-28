#include "lyrs.h"
#include "arrs.h"
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