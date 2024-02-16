#include "nn.h"
#include "activation.h"
#include "array.h"
#include "derivative.h"
#include "kernel.h"
#include "linalg.h"
#include "loss.h"
#include "mathematics.h"
#include "metric.h"
#include "random.h"

#include <chrono>
#include <random>
#include <string>

NN::NN(const std::vector<size_t> &layers, const float learning_rate)
{
    this->layers = layers;
    this->learning_rate = learning_rate;
}

void NN::Train(const Tensor &x_train, const Tensor &y_train, const Tensor &x_val, const Tensor &y_val)
{
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    size_t rd_num;
    std::random_device rd;
    std::vector<std::string> buffer;
    Tensor x_shuffled;
    Tensor y_shuffled;
    Tensor x_batch;
    Tensor y_batch;
    std::vector<Tensor> activations;
    std::vector<Tensor> activations_val;
    std::vector<Tensor> dloss_dlogits, dloss_dweights, dloss_dbiases;

    weights_biases = InitParameters();
    weights_biases_momentum = InitParameters();

    for (size_t i = 1; i <= epochs; ++i)
    {
        startTime = std::chrono::high_resolution_clock::now();

        if (i > 10 && i < 20)
            learning_rate = 0.009f;
        else if (i > 20 && i < 30)
            learning_rate = 0.005f;
        else
            learning_rate = 0.001f;

        rd_num = rd();
        x_shuffled = Shuffle(x_train, rd_num);
        y_shuffled = Shuffle(y_train, rd_num);

        // TODO: Fix not being able to change the batch size. For example, to 32 that's why it's stayed in 8.
        for (size_t j = 0; j < x_train.shape.front(); j += batch_size)
        {
            x_batch = Slice(x_shuffled, j, batch_size);
            y_batch = Slice(y_shuffled, j, batch_size);

            activations = ForwardPropagation(x_batch, weights_biases.first, weights_biases.second);

            size_t numLayers = layers.size() - 1;
            for (size_t k = numLayers; k > 0; --k)
            {
                // TODO: Find note for derivative that was deleted, and paste it here.
                if (k == numLayers)
                    dloss_dlogits.push_back(CategoricalCrossEntropyDerivative(y_batch, activations.back()));
                else
                    dloss_dlogits.push_back(MatMul(dloss_dlogits[(layers.size() - 2) - k],
                                                   Transpose(weights_biases.first[k]), Device::CPU) *
                                            ReluDerivative(activations[k - 1]));

                if (k == 1)
                    dloss_dweights.push_back(MatMul(Transpose(x_batch), dloss_dlogits[(numLayers)-k], Device::CPU));
                else
                    dloss_dweights.push_back(
                        MatMul(Transpose(activations[k - 2]), dloss_dlogits[(numLayers)-k], Device::CPU));

                dloss_dbiases.push_back(Sum(dloss_dlogits[(numLayers)-k], 0));

                dloss_dweights[(numLayers)-k] =
                    ClipByValue(dloss_dweights[(numLayers)-k], -gradient_clip_threshold, gradient_clip_threshold);
                dloss_dbiases[(numLayers)-k] =
                    ClipByValue(dloss_dbiases[(numLayers)-k], -gradient_clip_threshold, gradient_clip_threshold);

                weights_biases_momentum.first[k - 1] =
                    momentum * weights_biases_momentum.first[k - 1] - learning_rate * dloss_dweights[(numLayers)-k];
                weights_biases_momentum.second[k - 1] =
                    momentum * weights_biases_momentum.second[k - 1] - learning_rate * dloss_dbiases[(numLayers)-k];

                weights_biases.first[k - 1] += weights_biases_momentum.first[k - 1];
                weights_biases.second[k - 1] += weights_biases_momentum.second[k - 1];
            }

            dloss_dlogits.clear(), dloss_dweights.clear(), dloss_dbiases.clear();
        }

        activations_val = ForwardPropagation(x_val, weights_biases.first, weights_biases.second);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration);
        auto remainingMilliseconds = duration - seconds;

        // TODO: Add verbose arg to switch wheather to log metrics or not.
        buffer.push_back("Epoch " + std::to_string(i) + "/" + std::to_string(epochs) + "\n" +
                         std::to_string(seconds.count()) + "s " + std::to_string(remainingMilliseconds.count()) +
                         "ms/step - loss: " + std::to_string(CategoricalCrossEntropy(y_batch, activations.back())) +
                         " - accuracy: " + std::to_string(CategoricalAccuracy(y_batch, activations.back())));
        buffer.back() += " - val_loss: " + std::to_string(CategoricalCrossEntropy(y_val, activations_val.back())) +
                         " - val_accuracy: " + std::to_string(CategoricalAccuracy(y_val, activations_val.back()));

        if (i % 10 == 0)
        {
            for (const auto &message : buffer)
                std::cout << message << '\n';
            buffer.clear();
        }

        // static size_t epochs_without_improvement = 0;
        // static float best_val_loss = std::numeric_limits<float>::max();
        // float loss = CategoricalCrossEntropy(y_val, activations.back());

        // if (loss < best_val_loss) {
        //     best_val_loss = loss;
        //     epochs_without_improvement = 0;
        // } else {
        //     epochs_without_improvement += 1;
        // }

        // if (epochs_without_improvement >= patience) {
        //     std::cout << '\n' << "Early stopping at epoch " << i + 1 << " as validation loss did not improve for " <<
        //     patience << " epochs." << '\n'; break;
        // }
    }
}

void NN::Predict(const Tensor &x_test, const Tensor &y_test)
{
    std::vector<Tensor> activations = ForwardPropagation(x_test, weights_biases.first, weights_biases.second);

    std::cout << '\n';
    std::cout << "test loss: " << std::to_string(CategoricalCrossEntropy(y_test, activations.back()))
              << " - test accuracy: " << std::to_string(CategoricalAccuracy(y_test, activations.back()));
    std::cout << "\n\n";

    std::cout << activations.back() << "\n\n" << y_test << '\n';
}

std::pair<std::vector<Tensor>, std::vector<Tensor>> NN::InitParameters()
{
    std::vector<Tensor> weights;
    std::vector<Tensor> biases;

    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        weights.push_back(NormalDistribution({layers[i], layers[i + 1]}, 0.0f, 0.2f));
        biases.push_back(Zeros({1, layers[i + 1]}));
    }

    return std::make_pair(weights, biases);
}

std::vector<Tensor> NN::ForwardPropagation(const Tensor &input, const std::vector<Tensor> &weights,
                                           const std::vector<Tensor> &biases)
{
    std::vector<Tensor> activations;

    for (size_t i = 0; i < layers.size() - 1; ++i)
    {
        if (i == 0)
        {
            activations.push_back(Relu(MatMul(input, weights[i], Device::CPU) + biases[i], Device::CPU));
        }
        else
        {
            if (i == layers.size() - 2)
                activations.push_back(Softmax(MatMul(activations[i - 1], weights[i], Device::CPU) + biases[i]));
            else
                activations.push_back(
                    Relu(MatMul(activations[i - 1], weights[i], Device::CPU) + biases[i], Device::CPU));
        }
    }

    return activations;
}