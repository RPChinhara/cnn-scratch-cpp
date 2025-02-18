#include "acts.h"
#include "arrs.h"
#include "datasets.h"
#include "linalg.h"
#include "losses.h"
#include "lyrs.h"
#include "rand.h"

#include <chrono>

constexpr size_t vocab_size = 5000;
constexpr size_t seq_len    = 25;
constexpr size_t d_model    = 128; // NOTE: must be divisible by num_heads
constexpr size_t d_ff       = 512; // NOTE: often 4x larger than d_model
constexpr size_t num_heads  = 4;
constexpr size_t head_dim   = (num_heads == 1) ? d_model : d_model / num_heads;

std::vector<std::vector<tensor>> w_enc = {
    {glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim})}, // w_q, w_k, w_v
    {glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim})},
    {glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim})},
    {glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim})},
    {glorot_uniform({d_model, d_model})} // w_o
};

std::vector<std::vector<tensor>> w_dec_self = {
    {glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim})},
    {glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim})},
    {glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim})},
    {glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim})},
    {glorot_uniform({d_model, d_model})}
};

std::vector<std::vector<tensor>> w_dec_cross = {
    {glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim})},
    {glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim})},
    {glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim})},
    {glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim}), glorot_uniform({d_model, head_dim})},
    {glorot_uniform({d_model, d_model})}
};

// TODO: change to proper names only 1 and 2 should be used
tensor w1 = glorot_uniform({d_model, d_ff});
tensor w2 = glorot_uniform({d_ff, d_model});

tensor w3 = glorot_uniform({d_model, d_ff});
tensor w4 = glorot_uniform({d_ff, d_model});

tensor w_o = glorot_uniform({d_model, vocab_size});

tensor b1 = glorot_uniform({1, d_ff}); // NOTE: Could be (seq_len, d_ff), but it'd be inefficient for memory specially when the seq_len, d_model, and d_ff get much bigger.
tensor b2 = glorot_uniform({1, d_model});

tensor b3 = glorot_uniform({1, d_ff});
tensor b4 = glorot_uniform({1, d_model});

tensor b_o = glorot_uniform({1, vocab_size});

tensor encoder(const tensor& x) {
    size_t batch_size = x.shape.front();

    // NOTE: using postnorm, but there is prenorm as well
    tensor mha = multihead_attention(x, w_enc, seq_len, d_model, num_heads);
    tensor x1 = layer_normalization(x + mha);

    tensor x2 = zeros({batch_size, seq_len, d_model});
    for (size_t i = 0; i < batch_size; ++i) {
        tensor x1_mat = slice(x1, i * seq_len, seq_len);
        tensor x2_mat = matmul(relu(matmul(x1_mat, w1) + b1), w2) + b2;

        std::copy(x2_mat.elems, x2_mat.elems + x2_mat.size, x2.elems + i * x2_mat.size);
    }

    return layer_normalization(x1 + x2);
}

tensor decoder(const tensor& x, const tensor& encoder_output) {
    size_t batch_size = x.shape.front();

    tensor masked_mha = multihead_attention(x, w_dec_self, seq_len, d_model, num_heads, true);
    tensor x1 = layer_normalization(x + masked_mha);
    tensor cross_attention = multihead_cross_attention(x1, encoder_output, encoder_output, w_dec_cross, seq_len, d_model, num_heads);
    tensor x2 = layer_normalization(x1 + cross_attention);

    tensor x3 = zeros({batch_size, seq_len, d_model});
    for (size_t i = 0; i < batch_size; ++i) {
        tensor x2_mat = slice(x2, i * seq_len, seq_len);
        tensor x3_mat = matmul(relu(matmul(x2_mat, w3) + b3), w4) + b4;

        std::copy(x3_mat.elems, x3_mat.elems + x3_mat.size, x3.elems + i * x3_mat.size);
    }

    return layer_normalization(x2 + x3);
}

tensor train(const tensor& src_input, const tensor& tgt_input, const tensor& tgt_output) {
    constexpr size_t epochs = 5;
    constexpr float lr = 0.01f;
    constexpr float  batch_size = 32.0f;

    float num_samples = src_input.shape.front();
    const size_t num_batches = static_cast<size_t>(ceil(num_samples / batch_size));

    // NOTE: Embedding matrix is updated during backpropagation, similar to other model weights.
    auto embedding_lyr = embedding(vocab_size, d_model);
    auto positional_encoding_lyr = positional_encoding(seq_len, d_model);

    for (size_t i = 1; i <= epochs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        std::cout << "Epoch " << i << "/" << epochs << "\n";

        float loss = 0.0f;

        // TODO: I have to process multiple batches simultaneously in order to speed up training lol That is why batach training is faster right?
        for (size_t j = 0; j < num_batches; ++j) {
            size_t start_idx = j * batch_size;
            size_t end_idx = std::min(start_idx + batch_size, num_samples);

            tensor src_input_batch = slice(src_input, start_idx, end_idx - start_idx);
            tensor tgt_input_batch = slice(tgt_input, start_idx, end_idx - start_idx);
            tensor tgt_output_batch = slice(tgt_output, start_idx, end_idx - start_idx);

            tensor src_token_embeddings = embedding_lyr.adapt(src_input_batch);
            tensor src_positional_embeddings = positional_encoding_lyr.adapt(src_token_embeddings);

            tensor tgt_token_embeddings = embedding_lyr.adapt(tgt_input_batch);
            tensor tgt_positional_embeddings = positional_encoding_lyr.adapt(tgt_token_embeddings);

            // TODO: I run these functions simultaneously?
            tensor enc_output = encoder(src_positional_embeddings);
            tensor dec_output = decoder(tgt_positional_embeddings, enc_output); // (32, 25, 128)

            size_t batch_size = dec_output.shape.front();

            tensor probs = zeros({batch_size, seq_len, vocab_size});

            for (size_t k = 0; k < batch_size; ++k) {
                tensor dec_output_mat = slice(dec_output, k * seq_len, seq_len); // (25, 128)
                tensor logits_mat = matmul(dec_output_mat, w_o) + b_o; // TODO: Why it's called logits?
                tensor probs_mat = softmax(logits_mat);

                std::copy(probs_mat.elems, probs_mat.elems + probs_mat.size, probs.elems + k * probs_mat.size);
            }

            tgt_output_batch.reshape({batch_size * seq_len});
            probs.reshape({batch_size * seq_len, vocab_size});

            loss = sparse_categorical_cross_entropy(tgt_output_batch, probs);

            // Backpropagation

            if (j == num_batches - 1)
                std::cout << "\r\033[K";
            else
                std::cout << "\r\033[K" << j + 1 << "/" << num_batches << " - loss: " << loss << std::flush;
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::cout << num_batches << "/" << num_batches << " - " << std::chrono::duration<double>(end - start).count() << "s/step - loss: " << loss << "\n";
    }

    return tensor();
}

float evaluate(const tensor& x_test, const tensor& y_test) {
    return 0.0f;
}

tensor predict(const tensor& x_test, const tensor& y_test) {
    // You generate tokens one at a time (autoregressively).
    // At each step, you take the argmax of the last token's probability distribution (from softmax) and feed it back as the next input.
    return tensor();
}

int main() {
    auto data = load_daily_dialog();

    std::vector<std::string> vocab;
    vocab.reserve(data[0].size());

    for (const auto& str : data[0])
        vocab.emplace_back("<SOS> " + str + " <EOS>");

    // OPTIMIZE: Should I make vocabulary using Wikipedia, Common Crawl, OpenWebText, and ArXiv Papers or use pretrained ones such as BERT Vocabulary, GPT-2 Vocabulary.
    // TODO: I may need to use subword tokenizers for better results. I'm using a simple tokenizer.
    text_vectorizer vectorizer(vocab_size, seq_len);
    vectorizer.build_vocab(vocab);

    tensor src_input = vectorizer.adapt(data[0]);
    tensor tgt_input = vectorizer.adapt(data[1]);
    tensor tgt_output = vectorizer.adapt(data[2]);

    // TODO: put back to 0.2f
    auto src_input_train_test = split(src_input, 0.001f);   // (88821, 25), (89, 25)
    auto tgt_input_train_test = split(tgt_input, 0.001f);   // (88821, 25), (89, 25)
    auto tgt_output_train_test = split(tgt_output, 0.001f); // (88821, 25), (89, 25)

    train(src_input_train_test.second, tgt_input_train_test.second, tgt_output_train_test.second);

    // TODO: Combine evaluate() and predict()?
    // std::cout << "Test loss: " << evaluate(input_token_train_test.second, target_token_train_test.second) << "\n\n";

    // tensor test_predictions = predict(input_token_train_test.second, target_token_train_test.second);

    return 0;
}