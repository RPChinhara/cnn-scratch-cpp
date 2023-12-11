#include "nn.h"
#include "activation.h"
#include "array.h"
#include "derivative.h"
#include "linalg.h"
#include "mathematics.h"
#include "random.h"

#include <random>
#include <string>

NN::NN(const std::vector<size_t>& layers, float learning_rate)
{
    this->layers = layers;
    this->learning_rate = learning_rate;
}

void NN::Train(const Tensor& x_train, const Tensor& y_train, const Tensor& x_val, const Tensor& y_val)
{
    std::vector<std::string> buffer;
    
    weights_biases = InitParameters();
    weights_biases_momentum = InitParameters();

    for (size_t i = 1; i <= epochs; ++i) {
        if (i > 10 && i < 20)      learning_rate = 0.009f;
        else if (i > 20 && i < 30) learning_rate = 0.005f;
        else                       learning_rate = 0.001f;

        std::random_device rd;
        auto rd_num = rd();
        Tensor x_shuffled = Shuffle(x_train, rd_num);
        Tensor y_shuffled = Shuffle(y_train, rd_num);

        Tensor y_batch;
        TensorArray output;

        for (size_t j = 0; j < x_train.shape.front(); j += batch_size) {
            Tensor x_batch = Slice(x_shuffled, j, batch_size);
            y_batch = Slice(y_shuffled, j, batch_size);

            output = ForwardPropagation(x_batch, weights_biases.first, weights_biases.second);
            
            std::vector<Tensor> dloss_dlogits, dloss_dweights, dloss_dbiases;

            for (size_t k = layers.size() - 1; k > 0; --k) {
                if (k == layers.size() - 1)
                    dloss_dlogits.push_back(PrimeCategoricalCrossEntropy(y_batch, output.back()));
                else
                    dloss_dlogits.push_back(MatMul(dloss_dlogits[(layers.size() - 2) - k], Transpose(weights_biases.first[k])) * PrimeRelu(output[k - 1]));
            }

            for (size_t k = layers.size() - 1; k > 0; --k) {
                if (k == 1)
                    dloss_dweights.push_back(MatMul(Transpose(x_batch), dloss_dlogits[(layers.size() - 1) - k]));
                else
                    dloss_dweights.push_back(MatMul(Transpose(output[k - 2]), dloss_dlogits[(layers.size() - 1) - k]));
            }

            for (size_t k = 0; k < layers.size() - 1; ++k)
                dloss_dbiases.push_back(Sum(dloss_dlogits[k], 0));

            for (size_t k = 0; k < layers.size() - 1; ++k) {
                dloss_dweights[k] = ClipByValue(dloss_dweights[k], -gradient_clip_threshold, gradient_clip_threshold);
                dloss_dbiases[k] = ClipByValue(dloss_dbiases[k], -gradient_clip_threshold, gradient_clip_threshold);
            }

            for (int k = layers.size() - 2; k >= 0; --k) {
                weights_biases_momentum.first[k] = momentum * weights_biases_momentum.first[k] - learning_rate * dloss_dweights[(layers.size() - 2) - k];
                weights_biases_momentum.second[k] = momentum * weights_biases_momentum.second[k] - learning_rate * dloss_dbiases[(layers.size() - 2) - k];

                weights_biases.first[k] += weights_biases_momentum.first[k];
                weights_biases.second[k] += weights_biases_momentum.second[k];
            }
        }
        
        buffer.push_back("Epoch " + std::to_string(i) + "/" + std::to_string(epochs) + "\nloss: " + std::to_string(CategoricalCrossEntropy(y_batch, output.back())) + " - accuracy: " + std::to_string(CategoricalAccuracy(y_batch, output.back())));

        output = ForwardPropagation(x_val, weights_biases.first, weights_biases.second);

        buffer.back() += " - val_loss: " + std::to_string(CategoricalCrossEntropy(y_val, output.back())) + " - val_accuracy: " + std::to_string(CategoricalAccuracy(y_val, output.back()));

        if (i % 10 == 0) {
            for (const auto& message : buffer)
                std::cout << message << std::endl;
            buffer.clear();
        }

        // static size_t epochs_without_improvement = 0;
        // static float best_val_loss = std::numeric_limits<float>::max();
        // float loss = CategoricalCrossEntropy(y_val, output.back());

        // if (loss < best_val_loss) {
        //     best_val_loss = loss;
        //     epochs_without_improvement = 0;
        // } else {
        //     epochs_without_improvement += 1;
        // }

        // if (epochs_without_improvement >= patience) {
        //     std::cout << std::endl << "Early stopping at epoch " << i + 1 << " as validation loss did not improve for " << patience << " epochs." << std::endl;
        //     break;
        // }
    }
}

void NN::Predict(const Tensor& x_test, const Tensor& y_test)
{
    auto output = ForwardPropagation(x_test, weights_biases.first, weights_biases.second);

    std::cout << std::endl;
    std::cout << "test loss: " << CategoricalCrossEntropy(y_test, output.back()) << " - test accuracy: " << CategoricalAccuracy(y_test, output.back());
    std::cout << std::endl << std::endl;

    std::cout << output.back() << std::endl << std::endl << y_test << std::endl;
}

TensorArray NN::ForwardPropagation(const Tensor& input, const TensorArray& weights, const TensorArray& biases)
{
    TensorArray logits;
    TensorArray activations;

    for (size_t i = 0; i < layers.size() - 1; ++i) {
        if (i == 0) {
            logits.push_back((MatMul(input, weights[i]) + biases[i]));
            activations.push_back((Relu(logits[i])));
        } else {
            logits.push_back((MatMul(activations[i - 1], weights[i]) + biases[i]));
            if (i == 1)
                activations.push_back((Softmax(logits[i])));
        }
    }

    return activations;
}

std::pair<TensorArray, TensorArray> NN::InitParameters()
{
    TensorArray weights;
    TensorArray biases;

    for (size_t i = 0; i < layers.size() - 1; ++i) {
        weights.push_back(NormalDistribution({ layers[i], layers[i + 1] }, 0.0f, 0.2f));
        biases.push_back(Zeros({ 1, layers[i + 1] }));
    }

    return std::make_pair(weights, biases);
}