#include "lyrs.h"
#include "acts.h"
#include "arrs.h"
#include "linalg.h"
#include "rand.h"
#include "strings.h"

#include <random>
#include <unordered_map>

void min_max_scaler::fit(const tensor& data) {
    data_min = min(data);
    data_max = max(data, 0);
}

tensor min_max_scaler::transform(const tensor& data) {
    return (data - data_min) / (data_max - data_min);
}

tensor min_max_scaler::inverse_transform(const tensor& scaled_data) {
    return scaled_data * (data_max - data_min) + data_min;
}

void text_vectorizer::build_vocab(const std::vector<std::string>& data) {
    std::unordered_map<std::string, float> vocab_map;

    for (auto text : data) {
        auto tokens = tokenizer(text);

        for (auto token : tokens) {
            // TODO: I think I don't need these preprocessing sicne each dataset have different characters, it's so uncertain what to prepreprocess (remove). Best practice is to handle it in each load dataset functions e.g., load_daily_dialog(). text_vectorization() in TF remove all punctuations which is insane. All important ones like ?, !, and . as well.
            // token = lower(token);
            // token = regex_replace(token, "[\".,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", "");

            if (vocab_map.find(token) != vocab_map.end())
                vocab_map[token] += 1.0f;
            else
                vocab_map.insert(std::pair<std::string, float>(token, 1.0f));
        }
    }

    vocab_vec.assign(vocab_map.begin(), vocab_map.end());

    std::sort(vocab_vec.begin(), vocab_vec.end(), [](const std::pair<std::string, float> &a, const std::pair<std::string, float> &b) {
        if (a.second != b.second)
            return a.second > b.second;
        else
            return a.first > b.first;
    });

    vocab_vec.insert(vocab_vec.begin(), std::pair<std::string, float>("[UNK]", 1.0f));
    vocab_vec.insert(vocab_vec.begin(), std::pair<std::string, float>("", 0.0f));

    // NOTE: this will log first 50 vacabs in the list
    for (size_t i = 0; i < vocab_vec.size(); ++i)
      std::cout << vocab_vec[i].first << " " << vocab_vec[i].second << "\n";
}

tensor text_vectorizer::vectorize(const std::vector<std::string>& input) {
    tensor t_new = zeros({input.size(), seq_len});

    size_t idx = 0;
    const float oov_token = vocab_vec[1].second;

    // NOTE: In TensorFlow, the max_tokens (or vocabulary size) is max_tokens - 2 when output_mode == "int", because 0 is reserved for padding tokens and 1 is reserved for OOV (out-of-vocabulary) tokens.
    if (vocab_size > vocab_vec.size())
    vocab_size = vocab_vec.size();

    for (auto i = 0; i < input.size(); ++i) {
        auto words = tokenizer(input[i]);

        if (i != 0)
            idx = i * seq_len;

        size_t words_processed = 0;

        for (auto word : words) {
            ++words_processed;

            // TODO: I think I don't need these preprocessing sicne each dataset have different characters, it's so uncertain what to prepreprocess (remove). Best practice is to handle it in each load dataset functions e.g., load_daily_dialog(). text_vectorization() in TF remove all punctuations which is insane. All important ones like ?, !, and . as well.
            // word = lower(word);
            // word = regex_replace(word, "[\".,!?#$%&()*+/:;<=>@\\[\\]\\^_`{|}~\\\\-]", "");

            bool found = false;

            for (auto k = 0; k < vocab_size; ++k) {
                if (word == vocab_vec[k].first) {
                    t_new[idx] = k;
                    found = true;
                    break;
                }
            }

            if (!found)
                t_new[idx] = oov_token;

            if (words_processed == seq_len)
              break;

            ++idx;
        }
    }

    return t_new;
}

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
        float pos = static_cast<float>(i);
        for (size_t j = 0; j < dim / 2; ++j) {
            float denominator = pow(10000.0f, 2.0f * j / dim);
            pe(i, 2 * j) = sin(pos / denominator);
            pe(i, 2 * j + 1) = cos(pos / denominator);
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

        // TODO: Use std::copy(ffn.elems, ffn.elems + ffn.size, ffn_output.elems + i * ffn.size);
        for (size_t j = 0; j < output.size; ++j)
            outputs[i * output.size + j] = output[j];
    }

    return outputs;
}